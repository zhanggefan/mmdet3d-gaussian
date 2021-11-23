import torch
import torch.nn as nn
from torch.autograd import Function
from . import vsa_utils


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int):
        """
        Args:
            ctx:
            xyz: (B, N, 3) where N > npoint
            npoint: int, number of features in the sampled set

        Returns:
            output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = xyz.new_empty((B, npoint), dtype=torch.int32)
        temp = xyz.new_full((B, N), 1e10)

        vsa_utils.furthest_point_sampling(xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply
