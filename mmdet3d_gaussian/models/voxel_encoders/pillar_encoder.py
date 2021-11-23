from .utils import PointVoxelStatsCalculator
from ...ops import Scatter
from mmcv.cnn import build_norm_layer
from mmdet3d.models.builder import VOXEL_ENCODERS
from .utils import PFNLayer, get_paddings_indicator
from mmcv.runner import force_fp32, auto_fp16
import torch
from torch import nn


@VOXEL_ENCODERS.register_module(name='PillarFeatureNet', force=True)
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool, optional): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64,),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=False):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                        coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                        self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                        coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                        self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                        coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                        self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                        coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                        self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                        coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                        self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                        coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                        self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()


@VOXEL_ENCODERS.register_module(name='DynamicPillarFeatureNet', force=True)
class DynamicPillarFeatureNet(nn.Module):

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64,),
                 with_cluster_center=False,
                 with_cluster_center_offset=True,
                 with_covariance=False,
                 with_voxel_center=False,
                 with_voxel_point_count=False,
                 with_voxel_center_offset=True,
                 voxel_size=[(0.2, 0.2, 4)],
                 point_cloud_range=[(0, -40, -3, 70.4, 40, 1)],
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 reduce_op='max',
                 **kwargs):
        super(DynamicPillarFeatureNet, self).__init__()

        self.stats_cal = PointVoxelStatsCalculator(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            with_cluster_center=with_cluster_center,
            with_cluster_center_offset=with_cluster_center_offset,
            with_covariance=with_covariance,
            with_voxel_center=with_voxel_center,
            with_voxel_point_count=with_voxel_point_count,
            with_voxel_center_offset=with_voxel_center_offset)

        self.fp16_enabled = False

        in_channels = in_channels - 3 + self.stats_cal.out_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.reduce_op = reduce_op

    @force_fp32(apply_to=('features', 'points', 'img_feats'))
    def forward(self, features, coors, points=None, img_feats=None,
                img_metas=None):
        scatter = Scatter(coors)
        features = [self.stats_cal(features[:, :3], scatter), features[:, 3:]]
        features = torch.cat(features, dim=-1)

        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = scatter.reduce(point_feats,
                                                      self.reduce_op)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = scatter.mapback(voxel_feats)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        return voxel_feats, voxel_coors
