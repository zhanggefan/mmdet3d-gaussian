from mmdet3d.models.voxel_encoders.utils import *
import torch
from torch import nn
from ...ops import Scatter


class PointVoxelStatsCalculator(nn.Module):
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 with_cluster_center=True,
                 with_cluster_center_offset=True,
                 with_covariance=True,
                 with_voxel_center=True,
                 with_voxel_point_count=True,
                 with_voxel_center_offset=True):
        super(PointVoxelStatsCalculator, self).__init__()
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.with_cluster_center = with_cluster_center
        self.with_cluster_center_offset = with_cluster_center_offset
        self.with_covariance = with_covariance
        self.with_voxel_center = with_voxel_center
        self.with_voxel_center_offset = with_voxel_center_offset
        self.with_voxel_point_count = with_voxel_point_count

    @property
    def out_channels(self):
        _channels = 3
        if self.with_cluster_center:
            _channels += 3
        if self.with_cluster_center_offset:
            _channels += 3
        if self.with_covariance:
            _channels += 9
        if self.with_voxel_center:
            _channels += 3
        if self.with_voxel_center_offset:
            _channels += 3
        if self.with_voxel_point_count:
            _channels += 1
        return _channels

    def forward(self, pts_xyz: torch.Tensor, scatter: Scatter):
        assert pts_xyz.size(-1) == 3, 'feature tensor of shape (N, 3) ' \
                                      f'is expected, but got {pts_xyz.shape}'
        pts_coors = scatter.pts_coors

        if (self.with_cluster_center
                or self.with_cluster_center_offset
                or self.with_covariance):
            pts_cxyz = scatter.reduce_mapback(pts_xyz, reduce_op='mean')
            if self.with_cluster_center_offset or self.with_covariance:
                pts_diff_cxyz = pts_xyz - pts_cxyz
                if self.with_covariance:
                    pts_outprod_cxyz = (
                            pts_diff_cxyz[:, None, :] * pts_diff_cxyz[:, :,
                                                        None]).reshape(-1, 9)
                    pts_cov_cxyz = scatter.reduce_mapback(
                        pts_outprod_cxyz, reduce_op='mean')

        if self.with_voxel_center or self.with_voxel_center_offset:
            pts_coors_x = pts_coors[..., -1] * self.vx + self.x_offset
            pts_coors_y = pts_coors[..., -2] * self.vy + self.y_offset
            pts_coors_z = pts_coors[..., -3] * self.vz + self.z_offset
            pts_coors_xyz = torch.stack(
                (pts_coors_x, pts_coors_y, pts_coors_z), dim=-1)

        features_ls = [pts_xyz]
        if self.with_cluster_center:
            features_ls.append(pts_cxyz)
        if self.with_cluster_center_offset:
            features_ls.append(pts_diff_cxyz)
        if self.with_covariance:
            features_ls.append(pts_cov_cxyz)
        if self.with_voxel_center:
            features_ls.append(pts_coors_xyz)
        if self.with_voxel_center_offset:
            features_ls.append(pts_xyz - pts_coors_xyz)
        if self.with_voxel_point_count:
            pts_pts_counts = scatter.mapback(
                scatter.voxel_pts_counts.unsqueeze(-1))
            features_ls.append(pts_pts_counts)

        return torch.cat(features_ls, dim=-1)
