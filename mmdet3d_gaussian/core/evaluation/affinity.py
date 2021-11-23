from ...ops.eval.eval_utils import trans_bev, iou_3d, iou_bev
from .builder import EVAL_AFFINITYCALS


@EVAL_AFFINITYCALS.register_module()
class LidarCenterTransBEV:
    LARGER_CLOSER = False

    def __call__(self, det_bboxes, gt_bboxes, gt_iscrowd=None):
        assert gt_iscrowd is None, 'Does not support crowd annotation yet'
        return trans_bev(det_bboxes, gt_bboxes)


@EVAL_AFFINITYCALS.register_module()
class LidarIOU3D:
    LARGER_CLOSER = True

    def __init__(self, z_offset=0.5):
        self.z_offset = z_offset

    def __call__(self, det_bboxes, gt_bboxes, gt_iscrowd=None):
        assert gt_iscrowd is None, 'Does not support crowd annotation yet'
        return iou_3d(det_bboxes, gt_bboxes, self.z_offset)


@EVAL_AFFINITYCALS.register_module()
class LidarIOUBEV:
    LARGER_CLOSER = True

    def __call__(self, det_bboxes, gt_bboxes, gt_iscrowd=None):
        assert gt_iscrowd is None, 'Does not support crowd annotation yet'
        return iou_bev(det_bboxes, gt_bboxes)
