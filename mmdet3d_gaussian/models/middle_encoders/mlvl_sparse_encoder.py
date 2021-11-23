# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from mmdet3d.models.builder import MIDDLE_ENCODERS
from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder


@MIDDLE_ENCODERS.register_module()
class MlvlSparseEncoder(SparseEncoder):
    @auto_fp16(apply_to=('voxel_features',))
    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        return dict(encode_features=encode_features,
                    spatial_features=spatial_features)
