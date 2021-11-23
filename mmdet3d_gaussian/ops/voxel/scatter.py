import torch
from torch import nn
from torch.autograd import Function

from .voxel_utils import (dynamic_point_to_voxel_backward,
                          dynamic_point_to_voxel_scatter_index,
                          dynamic_point_to_voxel_scatter_reduce)


class _scatter_index(Function):
    @staticmethod
    def forward(ctx, coors):
        (voxel_coors,
         point2voxel_map,
         voxel_points_count) = dynamic_point_to_voxel_scatter_index(coors)
        ctx.mark_non_differentiable(voxel_coors, point2voxel_map,
                                    voxel_points_count)
        return voxel_coors, point2voxel_map, voxel_points_count

    @staticmethod
    def backward(ctx, grad_voxel_coors=None,
                 grad_point2voxel_map=None, grad_voxel_points_count=None):
        return None


scatter_index = _scatter_index.apply


class _scatter_reduce(Function):

    @staticmethod
    def forward(ctx, feats, point2voxel_map, voxel_points_count,
                reduce_type='max'):
        """convert kitti points(N, >=3) to voxels.

        Args:
            feats: [N, C] float tensor. points features to be reduced
                into voxels.
            coors: [N, ndim] int tensor. corresponding voxel coordinates
                (specifically multi-dim voxel index) of each points.
            reduce_type: str. reduce op. support 'max', 'sum' and 'mean'
        Returns:
            tuple
            voxel_feats: [M, C] float tensor. reduced features. input features
                that shares the same voxel coordinates are reduced to one row
            coordinates: [M, ndim] int tensor, voxel coordinates.
        """
        voxel_feats = dynamic_point_to_voxel_scatter_reduce(feats,
                                                            point2voxel_map,
                                                            voxel_points_count,
                                                            reduce_type)
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map,
                              voxel_points_count)
        ctx.mark_non_differentiable(point2voxel_map, voxel_points_count)
        return voxel_feats

    @staticmethod
    def backward(ctx, grad_voxel_feats):
        (feats, voxel_feats, point2voxel_map,
         voxel_points_count) = ctx.saved_tensors
        grad_feats = torch.zeros_like(feats)
        # TODO: whether to use index put or use cuda_backward
        # To use index put, need point to voxel index
        dynamic_point_to_voxel_backward(grad_feats,
                                        grad_voxel_feats.contiguous(), feats,
                                        voxel_feats, point2voxel_map,
                                        voxel_points_count, ctx.reduce_type)
        return grad_feats, None, None, None


scatter_reduce = _scatter_reduce.apply


class Scatter(object):
    def __init__(self, coors):
        self._pts_coors = coors
        if coors.numel() == 0:
            self._batch_size = None if coors.size(-1) == 3 else 1
            self.voxel_coors = coors.clone().detach()
            self.pts_voxel_maps = coors.new_empty((0,), dtype=torch.int32)
            self.voxel_pts_counts = coors.new_empty((0,), dtype=torch.int32)
            return

        if coors.size(-1) == 3:
            self._batch_size = None
            (voxel_coors, pts_voxel_maps, voxel_pts_counts) = scatter_index(
                coors.contiguous())
        else:
            batch_size = coors[:, 0].max().item() + 1
            self._batch_size = batch_size
            previous_voxels = 0
            pts_voxel_maps = coors.new_full(
                (coors.size(0),), -1, dtype=torch.int32)
            voxel_pts_counts = []
            voxel_coors = []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                (voxel_coor, pts_voxel_map, voxel_pts_count) = scatter_index(
                    coors[inds][:, 1:].contiguous())

                pts_voxel_map[pts_voxel_map.ge(0)] += previous_voxels
                pts_voxel_maps[inds] = pts_voxel_map
                previous_voxels += voxel_coor.size(0)

                voxel_pts_counts.append(voxel_pts_count)

                coor_pad = nn.functional.pad(
                    voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)

            voxel_coors = torch.cat(voxel_coors, dim=0)
            voxel_pts_counts = torch.cat(voxel_pts_counts, dim=0)

        self.voxel_coors = voxel_coors
        self.pts_voxel_maps = pts_voxel_maps
        self.voxel_pts_counts = voxel_pts_counts

    @property
    def pts_coors(self):
        return self._pts_coors

    @property
    def batch_size(self):
        return self._batch_size

    def mapback(self, voxel_feats, default_feat=0):
        invalid_mask = self.pts_voxel_maps.lt(0)
        point_feats = voxel_feats[self.pts_voxel_maps.clamp(min=0).long()]
        point_feats[invalid_mask] = default_feat
        return point_feats

    def reduce(self, pts_feats, reduce_op):
        assert reduce_op in ['max', 'mean', 'sum'], \
            'For the arg "reduce", only "max", "mean" and "sum"' \
            f' are supported but got {reduce_op}'
        voxel_feats = scatter_reduce(pts_feats.contiguous(),
                                     self.pts_voxel_maps,
                                     self.voxel_pts_counts, reduce_op)
        return voxel_feats, self.voxel_coors

    def reduce_mapback(self, pts_feats, reduce_op, default_feat=0):
        voxel_feats, _ = self.reduce(pts_feats, reduce_op)
        return self.mapback(voxel_feats, default_feat)
