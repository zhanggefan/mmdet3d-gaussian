import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from mmcv.cnn.bricks import build_norm_layer, is_norm
from mmdet3d.models.builder import MIDDLE_ENCODERS
from ...ops.vsa import QueryAndGroup, furthest_point_sample


class GuidedSAModuleMSG(nn.Module):

    def __init__(self, in_channels: int, radii: List[float],
                 nsamples: List[int], mlps: List[List[int]],
                 use_xyz: bool = True, pool_method='max',
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01)):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max / avg
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            cin = in_channels
            if use_xyz:
                cin += 3
            radius = radii[i]
            nsample = nsamples[i]
            mlp_spec = mlps[i]

            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz))

            shared_mlps = []
            for cout in mlp_spec:
                shared_mlps.extend([
                    nn.Conv1d(cin, cout, 1, bias=False),
                    build_norm_layer(norm_cfg, cout)[1],
                    nn.ReLU()])
                cin = cout
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if is_norm(m):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, Cin, nsample)
            new_features = self.mlps[k](new_features)
            # (M1 + M2 ..., Cout, nsample)

            if self.pool_method == 'max':
                new_features = new_features.max(-1).values
            elif self.pool_method == 'avg':
                new_features = new_features.mean(-1)
            else:
                raise NotImplementedError
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


@MIDDLE_ENCODERS.register_module()
class VoxelSetAbstraction(nn.Module):
    def __init__(self, num_keypoints, out_channels, voxel_size,
                 point_cloud_range, voxel_sa_configs, rawpoint_sa_config=None,
                 bev_sa_config=None, voxel_center_as_source=False,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 voxel_center_align='half',
                 debug=False):
        super().__init__()
        self.debug = debug
        assert voxel_center_align in ('half', 'halfmin')
        self.voxel_center_align = voxel_center_align
        self.num_keypoints = num_keypoints
        self.out_channels = out_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_center_as_source = voxel_center_as_source

        self.voxel_sa_configs = voxel_sa_configs
        self.rawpoint_sa_config = rawpoint_sa_config
        self.bev_sa_config = bev_sa_config

        self.voxel_sa_layers = nn.ModuleList()

        gathered_channels = 0
        for voxel_sa_config in voxel_sa_configs:
            cur_layer = GuidedSAModuleMSG(
                in_channels=voxel_sa_config.in_channels,
                radii=voxel_sa_config.pool_radius,
                nsamples=voxel_sa_config.samples,
                mlps=voxel_sa_config.mlps,
                use_xyz=True,
                pool_method='max',
                norm_cfg=norm_cfg)
            self.voxel_sa_layers.append(cur_layer)
            gathered_channels += sum([x[-1] for x in voxel_sa_config.mlps])

        self.bev_sa = (bev_sa_config is not None)
        if bev_sa_config is not None:
            self.bev_sa = True
            gathered_channels += bev_sa_config.in_channels

        self.rawpoints_sa = (rawpoint_sa_config is not None)
        if rawpoint_sa_config is not None:
            self.rawpoints_sa_layer = GuidedSAModuleMSG(
                in_channels=rawpoint_sa_config.in_channels,
                radii=rawpoint_sa_config.pool_radius,
                nsamples=rawpoint_sa_config.samples,
                mlps=rawpoint_sa_config.mlps,
                use_xyz=True,
                pool_method='max',
                norm_cfg=norm_cfg)
            gathered_channels += sum([x[-1] for x in rawpoint_sa_config.mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(gathered_channels, out_channels, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU())

    def interpolate_from_bev_features(self, keypoints, bev_features,
                                      scale_factor):
        _, _, y_grid, x_grid = bev_features.shape

        voxel_size_xy = keypoints.new_tensor(self.voxel_size[:2])

        bev_tl_grid_cxy = keypoints.new_tensor(self.point_cloud_range[:2])
        bev_br_grid_cxy = keypoints.new_tensor(self.point_cloud_range[3:5])
        if self.voxel_center_align == 'half':
            bev_tl_grid_cxy.add_(0.5 * voxel_size_xy * scale_factor)
            bev_br_grid_cxy.sub_(0.5 * voxel_size_xy * scale_factor)
        elif self.voxel_center_align == 'halfmin':
            bev_tl_grid_cxy.add_(0.5 * voxel_size_xy)
            bev_br_grid_cxy.sub_(voxel_size_xy * (scale_factor - 0.5))

        xy = keypoints[..., :2]

        grid_sample_xy = (xy - bev_tl_grid_cxy[None, None, :]) / (
            (bev_br_grid_cxy - bev_tl_grid_cxy)[None, None, :])

        grid_sample_xy = (grid_sample_xy * 2 - 1).unsqueeze(1)
        point_bev_features = F.grid_sample(bev_features,
                                           grid=grid_sample_xy,
                                           align_corners=True)
        return point_bev_features.squeeze(2).permute(0, 2, 1).contiguous()

    def get_voxel_centers(self, coors, scale_factor):
        assert coors.shape[1] == 4
        voxel_centers = coors[:, [3, 2, 1]].float()  # (xyz)
        voxel_size = voxel_centers.new_tensor(self.voxel_size)
        pc_range_min = voxel_centers.new_tensor(self.point_cloud_range[:3])

        voxel_centers = voxel_centers * voxel_size * scale_factor + pc_range_min

        if self.voxel_center_align == 'half':
            voxel_centers.add_(0.5 * voxel_size * scale_factor)
        elif self.voxel_center_align == 'halfmin':
            voxel_centers.add_(0.5 * voxel_size)
        else:
            raise NotImplementedError
        return voxel_centers

    def get_sampled_points(self, points, coors):
        assert points is not None or coors is not None
        if self.voxel_center_as_source:
            _src_points = self.get_voxel_centers(coors=coors, scale_factor=1)
            batch_size = coors[-1, 0].item() + 1
            src_points = [_src_points[coors[:, 0] == b] for b in
                          range(batch_size)]
        else:
            src_points = [p[..., :3] for p in points]

        keypoints_list = []
        for points_to_sample in src_points:
            num_points = points_to_sample.shape[0]
            cur_pt_idxs = furthest_point_sample(
                points_to_sample.unsqueeze(dim=0).contiguous(),
                self.num_keypoints).long()[0]

            if num_points < self.num_keypoints:
                times = int(self.num_keypoints / num_points) + 1
                non_empty = cur_pt_idxs[:num_points]
                cur_pt_idxs = non_empty.repeat(times)[:self.num_keypoints]

            keypoints = points_to_sample[cur_pt_idxs]

            keypoints_list.append(keypoints)
            if self.debug:
                import open3d as o3d
                import numpy as np
                kpts = keypoints.cpu().detach().numpy().astype(np.float64)
                pts = points_to_sample.cpu().detach().numpy().astype(np.float64)

                kpc = o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(kpts))
                pc = o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(pts))

                kcolor = np.zeros_like(kpts)
                kcolor[..., 2] = 1
                kcolor = o3d.utility.Vector3dVector(kcolor)
                kpc.colors = kcolor

                color = np.zeros_like(pts)
                color[..., 0] = 1
                color = o3d.utility.Vector3dVector(color)
                pc.colors = color

                o3d.visualization.draw_geometries([pc, kpc])

        keypoints = torch.stack(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def forward(self, voxel_encode_features, points=None, coors=None,
                bev_encode_features=None):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(points, coors)

        point_features_list = []
        if self.bev_sa:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, bev_encode_features, self.bev_sa_config.scale_factor)
            point_features_list.append(point_bev_features)

        batch_size, num_keypoints, _ = keypoints.shape
        key_xyz = keypoints.view(-1, 3)
        key_xyz_batch_cnt = key_xyz.new_zeros(batch_size).int().fill_(
            num_keypoints)

        if self.rawpoints_sa:
            batch_points = torch.cat(points, dim=0)
            batch_cnt = [len(p) for p in points]
            xyz = batch_points[:, :3].contiguous()
            features = None
            if batch_points.size(1) > 0:
                features = batch_points[:, 3:].contiguous()
            xyz_batch_cnt = xyz.new_tensor(batch_cnt, dtype=torch.int32)

            pooled_points, pooled_features = self.rawpoints_sa_layer(
                xyz=xyz,
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=key_xyz,
                new_xyz_batch_cnt=key_xyz_batch_cnt,
                features=features,
            )
            point_features_list.append(
                pooled_features.view(batch_size, num_keypoints, -1))

        for k, voxel_sa_layer in enumerate(self.voxel_sa_layers):
            cur_coords = voxel_encode_features[k].indices
            xyz = self.get_voxel_centers(
                coors=cur_coords,
                scale_factor=self.voxel_sa_configs[k].scale_factor
            ).contiguous()
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = voxel_sa_layer(
                xyz=xyz,
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=key_xyz,
                new_xyz_batch_cnt=key_xyz_batch_cnt,
                features=voxel_encode_features[k].features.contiguous(),
            )
            point_features_list.append(
                pooled_features.view(batch_size, num_keypoints, -1))

        point_features = torch.cat(point_features_list, dim=2).view(
            batch_size * num_keypoints, -1)

        fusion_point_features = self.vsa_point_feature_fusion(point_features)

        bid = torch.arange(batch_size * num_keypoints,
                           device=keypoints.device) // num_keypoints
        key_bxyz = torch.cat((bid.to(key_xyz.dtype).unsqueeze(dim=-1),
                              key_xyz), dim=-1)

        return dict(keypoint_features=point_features,
                    fusion_keypoint_features=fusion_point_features,
                    keypoints=key_bxyz)
