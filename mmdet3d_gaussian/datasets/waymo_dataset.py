import mmcv
import torch
from mmdet.datasets import DATASETS
from .mem_util import SharedList
from mmdet3d.datasets import WaymoDataset


@DATASETS.register_module(name='WaymoDataset', force=True)
class WaymoDatasetRev(WaymoDataset):
    def load_annotations(self, ann_file):
        is_master = torch.cuda.current_device() == 0
        annotations = mmcv.load(ann_file) if is_master else None
        return SharedList(annotations)
