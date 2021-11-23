from mmdet.models import DETECTORS
from mmdet3d.models.detectors import (
    CenterPoint, VoxelNet, MVXFasterRCNN, DynamicVoxelNet)
from .utils import ShowResultMixin


@DETECTORS.register_module(name='CenterPoint', force=True)
class CenterPointRev(ShowResultMixin, CenterPoint):
    def forward_test(self, **kwargs):
        kwargs2 = {k: v for k, v in kwargs.items() if not k.startswith('gt_')}
        return super(CenterPoint, self).forward_test(**kwargs2)


@DETECTORS.register_module(name='VoxelNet', force=True)
class VoxelNetRev(ShowResultMixin, VoxelNet):
    def forward_test(self, **kwargs):
        kwargs2 = {k: v for k, v in kwargs.items() if not k.startswith('gt_')}
        return super(VoxelNetRev, self).forward_test(**kwargs2)


@DETECTORS.register_module(name='MVXFasterRCNN', force=True)
class VMVXFasterRCNNRev(ShowResultMixin, MVXFasterRCNN):
    def forward_test(self, **kwargs):
        kwargs2 = {k: v for k, v in kwargs.items() if not k.startswith('gt_')}
        return super(VMVXFasterRCNNRev, self).forward_test(**kwargs2)

@DETECTORS.register_module(name='DynamicVoxelNet', force=True)
class DynamicVoxelNetRev(ShowResultMixin, DynamicVoxelNet):
    def forward_test(self, **kwargs):
        kwargs2 = {k: v for k, v in kwargs.items() if not k.startswith('gt_')}
        return super(DynamicVoxelNetRev, self).forward_test(**kwargs2)
