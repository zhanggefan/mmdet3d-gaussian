from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset


@DATASETS.register_module()
class NuScenesDatasetRev(NuScenesDataset):
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE',
        'iou3d_err': 'mAIE'
    }
