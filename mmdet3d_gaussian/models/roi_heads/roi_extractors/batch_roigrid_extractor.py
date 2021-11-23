# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule

from ...middle_encoders.voxel_set_abstraction import GuidedSAModuleMSG
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet3d.core.bbox.structures import rotation_3d_in_axis


@ROI_EXTRACTORS.register_module()
class Batch3DRoIGridExtractor(BaseModule):
    def __init__(self, in_channels, pool_radius, samples, mlps, grid_size=6,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max', init_cfg=None):
        super(Batch3DRoIGridExtractor, self).__init__(init_cfg=init_cfg)
        self.roi_grid_pool_layer = GuidedSAModuleMSG(
            in_channels=in_channels,
            radii=pool_radius,
            nsamples=samples,
            mlps=mlps,
            use_xyz=True,
            pool_method=mode,
            norm_cfg=norm_cfg)
        self.grid_size = grid_size

    def forward(self, feats, coordinate, batch_inds, rois):
        batch_size = int(batch_inds.max()) + 1

        xyz = coordinate
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_inds == k).sum()

        rois_batch_inds = rois[:, 0].int()
        # (N1+N2+..., 6x6x6, 3)
        roi_grid = self.get_dense_grid_points(rois[:, 1:])

        new_xyz = roi_grid.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = ((rois_batch_inds == k).sum() *
                                    roi_grid.size(1))
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=feats.contiguous())  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.grid_size, self.grid_size, self.grid_size,
            pooled_features.shape[-1])
        # (BxN, 6, 6, 6, C)
        return pooled_features

    def get_dense_grid_points(self, rois):
        faked_features = rois.new_ones(
            (self.grid_size, self.grid_size, self.grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(rois.size(0), 1, 1).float()
        dense_idx = ((dense_idx + 0.5) / self.grid_size)
        dense_idx[..., :2] -= 0.5

        roi_ctr = rois[:, :3]
        roi_dim = rois[:, 3:6]
        roi_grid_points = dense_idx * roi_dim.view(-1, 1, 3)
        roi_grid_points = rotation_3d_in_axis(
            roi_grid_points, rois[:, 6], axis=2)
        roi_grid_points += roi_ctr.view(-1, 1, 3)

        return roi_grid_points
