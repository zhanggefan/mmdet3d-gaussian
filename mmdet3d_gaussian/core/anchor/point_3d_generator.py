import torch
from mmdet.core.anchor.builder import PRIOR_GENERATORS


@PRIOR_GENERATORS.register_module()
class Point3DRangeGenerator(object):
    def __init__(self, ranges, base=1):
        self.ranges = ranges
        self.base = base

    @property
    def num_levels(self):
        return len(self.ranges)

    def grid_anchors(self, featmap_sizes, device='cuda'):
        return self.grid_priors(featmap_sizes, device)

    def grid_priors(self, featmap_sizes, device='cuda'):
        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(featmap_sizes[i],
                                                   level_idx=i,
                                                   device=device)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 feature_size,
                                 level_idx,
                                 device='cuda'):
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(self.ranges[level_idx], device=device)
        y_centers = torch.linspace(
            anchor_range[1], anchor_range[3], feature_size[1], device=device)
        x_centers = torch.linspace(
            anchor_range[0], anchor_range[2], feature_size[2], device=device)
        grid_x_centers, grid_y_centers = torch.meshgrid(
            x_centers, y_centers)
        stride = self.base * (anchor_range[2] - anchor_range[0]) / (
            feature_size[2] - 1)
        stride = torch.full_like(grid_x_centers, stride)
        priors = torch.stack(
            (grid_x_centers, grid_y_centers, stride), dim=-1)
        priors = priors.permute((1, 0, 2)).contiguous()
        return priors
