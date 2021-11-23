import torch
from mmcv.cnn import (build_norm_layer, build_upsample_layer)
from mmcv.runner import force_fp32, auto_fp16
from torch import nn
import torch.nn.functional as F
import math
from ...ops import Scatter
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models.middle_encoders import PointPillarsScatter
from .utils import PointVoxelStatsCalculator


class SingleViewNet(nn.Module):
    def __init__(self, in_channels, feat_channels, voxel_size,
                 point_cloud_range,
                 norm1d_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 norm2d_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 reduce_op='max',
                 **kwargs):
        super(SingleViewNet, self).__init__()
        self.pointnet = nn.Sequential(
            nn.Linear(in_channels, feat_channels, bias=False),
            build_norm_layer(norm1d_cfg, feat_channels)[1],
            nn.ReLU(inplace=True))
        self.res1 = BasicBlock(inplanes=feat_channels, planes=feat_channels,
                               norm_cfg=norm2d_cfg)
        self.res2 = BasicBlock(inplanes=feat_channels, planes=feat_channels,
                               downsample=nn.Sequential(
                                   nn.Conv2d(feat_channels,
                                             feat_channels, 1, 2, bias=False),
                                   build_norm_layer(norm2d_cfg, feat_channels)[
                                       1]),
                               stride=2, norm_cfg=norm2d_cfg)
        self.res3 = BasicBlock(inplanes=feat_channels, planes=feat_channels,
                               downsample=nn.Sequential(
                                   nn.Conv2d(feat_channels,
                                             feat_channels, 1, 2, bias=False),
                                   build_norm_layer(norm2d_cfg, feat_channels)[
                                       1]),
                               stride=2, norm_cfg=norm2d_cfg)

        self.deconv2 = build_upsample_layer(
            upsample_cfg,
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=2,
            stride=2)
        self.deconv3 = build_upsample_layer(
            upsample_cfg,
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=4,
            stride=4)
        self.conv = nn.Conv2d(in_channels=3 * feat_channels,
                              out_channels=feat_channels, kernel_size=3,
                              padding=1)
        self.vs = voxel_size
        self.pcrange = point_cloud_range
        nx = math.ceil((self.pcrange[3] - self.pcrange[0]) / self.vs[0])
        ny = math.ceil((self.pcrange[4] - self.pcrange[1]) / self.vs[1])
        self.nx = nx
        self.ny = ny
        self.pillar_scatter = PointPillarsScatter(feat_channels, (ny, nx))
        self.feat_channels = feat_channels
        self.reduce_op = reduce_op
        self.fp16_enabled = False

    @auto_fp16(out_fp32=True)
    def forward_view(self, x):
        out1 = self.res1(x)
        out2 = self.res2(x)
        out3 = self.res3(out2)
        out2 = self.deconv2(out2)
        out3 = self.deconv3(out3)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.conv(out)

    def forward(self, pts_xyz: torch.Tensor, pts_features: torch.Tensor,
                scatter: Scatter):
        pts_x = self.pointnet(pts_features)
        voxel_x, voxel_coors = scatter.reduce(pts_x, self.reduce_op)
        x = self.pillar_scatter(voxel_x, voxel_coors, scatter.batch_size)

        out = self.forward_view(x)

        grid_x = (pts_xyz[:, 0] - self.pcrange[0]
                  ) / ((self.pcrange[3] - self.pcrange[0]) / 2) - 1
        grid_y = (pts_xyz[:, 1] - self.pcrange[1]
                  ) / ((self.pcrange[4] - self.pcrange[1]) / 2) - 1
        grid_xy = torch.stack((grid_x, grid_y), dim=-1)

        pts_out = pts_features.new_empty(
            (pts_features.size(0), self.feat_channels))
        for i in range(scatter.batch_size):
            b_idx = torch.where(scatter.pts_coors[:, 0] == i)
            b_grid_xy = grid_xy[b_idx]
            b_grid_xy = b_grid_xy[None, None, ...]
            b_out = out[i:(i + 1)]
            b_pts_out = F.grid_sample(
                b_out, b_grid_xy, mode='bilinear', padding_mode='zeros',
                align_corners=False)
            b_pts_out = b_pts_out.reshape(self.feat_channels, -1).permute(1, 0)
            pts_out[b_idx] = b_pts_out

        return pts_out


@VOXEL_ENCODERS.register_module()
class PillarMVFFeatureNet(nn.Module):
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
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=128,
                 with_cluster_center=True,
                 with_cluster_center_offset=True,
                 with_covariance=True,
                 with_voxel_center=True,
                 with_voxel_point_count=True,
                 with_voxel_center_offset=True,
                 voxel_size=[(0.2, 0.2, 4)],
                 point_cloud_range=[(0, -40, -3, 70.4, 40, 1)],
                 norm1d_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 norm2d_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 reduce_op='max'):
        super(PillarMVFFeatureNet, self).__init__()

        in_channels_pointnet1 = in_channels - 3  # channel count except xyz
        stats_cal = []
        for vsize, pcrange in zip(voxel_size, point_cloud_range):
            pvsc = PointVoxelStatsCalculator(
                voxel_size=vsize,
                point_cloud_range=pcrange,
                with_cluster_center=with_cluster_center,
                with_cluster_center_offset=with_cluster_center_offset,
                with_covariance=with_covariance,
                with_voxel_center=with_voxel_center,
                with_voxel_point_count=with_voxel_point_count,
                with_voxel_center_offset=with_voxel_center_offset)
            stats_cal.append(pvsc)
            in_channels_pointnet1 += pvsc.out_channels
        self.stats_cal = nn.ModuleList(stats_cal)

        view_nets = []
        in_channels_pointnet3 = feat_channels
        for vsize, pcrange in zip(voxel_size, point_cloud_range):
            view_nets.append(
                SingleViewNet(in_channels=feat_channels,
                              feat_channels=feat_channels,
                              voxel_size=vsize,
                              point_cloud_range=pcrange,
                              norm1d_cfg=norm1d_cfg,
                              norm2d_cfg=norm2d_cfg,
                              upsample_cfg=upsample_cfg,
                              reduce_op=reduce_op))
            in_channels_pointnet3 += feat_channels
        self.view_nets = nn.ModuleList(view_nets)

        self.pointnet1 = nn.Sequential(
            nn.Linear(in_channels_pointnet1, feat_channels, bias=False),
            build_norm_layer(norm1d_cfg, feat_channels)[1],
            nn.ReLU(inplace=True))
        self.pointnet2 = nn.Sequential(
            nn.Linear(feat_channels, feat_channels, bias=False),
            build_norm_layer(norm1d_cfg, feat_channels)[1],
            nn.ReLU(inplace=True))
        self.pointnet3 = nn.Sequential(
            nn.Linear(in_channels_pointnet3, feat_channels, bias=False),
            build_norm_layer(norm1d_cfg, feat_channels)[1],
            nn.ReLU(inplace=True))

        self.reduce_op = reduce_op
        self.fp16_enabled = False

    @force_fp32(apply_to=('multi_features', 'points', 'img_feats'))
    def forward(self, multi_features, multi_coors, points=None, img_feats=None,
                img_metas=None):
        """Forward function.

        Args:
            features (torch.Tensor): 
            multi_xyz (list[torch.Tensor]): List of point xyz
                from different coordinate system. Each in shape (N, 3).
            multi_coors (list[torch.Tensor]): List of coordinates of 
                each voxel from different coordinate system.

        Returns:
            torch.Tensor: Features of pillars.
        """
        invalid_flag = None
        for coors in multi_coors:
            _invalid_flag = coors[:, -3:].lt(0).all(dim=-1)
            if invalid_flag is None:
                invalid_flag = _invalid_flag
            else:
                invalid_flag.logical_or_(_invalid_flag)
        for coors in multi_coors:
            coors[invalid_flag][:, -3:] = -1
        scatters = [Scatter(coors) for coors in multi_coors]
        multi_xyz = [features[:, :3] for features in multi_features]
        features = multi_features[0][:, 3:]

        pts_feats = []
        for xyz, pvsc, scatter in zip(multi_xyz, self.stats_cal, scatters):
            pts_feats.append(pvsc(xyz, scatter))
        pts_feats.append(features)
        pts_feats = torch.cat(pts_feats, dim=-1)

        pts_feats = self.pointnet1(pts_feats)

        x_mvf = []
        for view_net, xyz, scatter in zip(self.view_nets, multi_xyz, scatters):
            x_mvf.append(view_net(xyz, pts_feats, scatter))

        x_pointwise = self.pointnet2(pts_feats)

        x_mvf.append(x_pointwise)
        x_mvf = torch.cat(x_mvf, dim=-1)
        x_mvf = self.pointnet3(x_mvf)

        return scatters[0].reduce(x_mvf, self.reduce_op)
