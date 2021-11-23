import numpy as np
from ...ops.eval.eval_utils import match_coco
from .builder import EVAL_MATCHERS


class BaseMatcher:
    def __init__(self, match_thrs, affinity_cost_negate=True):
        self._match_thrs = match_thrs
        self.negate = affinity_cost_negate

    @property
    def match_thrs(self):
        return self._match_thrs

    def __call__(self, affinity, gt_isignore=None, gt_iscrowd=None):
        if gt_iscrowd is None:
            gt_iscrowd = np.zeros(affinity.shape[1], dtype=np.bool)
        if gt_isignore is None:
            gt_isignore = np.zeros(affinity.shape[1], dtype=np.bool)
        if self.negate:
            return self.match(-affinity,
                              -np.array(self.match_thrs, np.float32),
                              gt_isignore, gt_iscrowd)
        else:
            return self.match(affinity, np.array(self.match_thrs, np.float32),
                              gt_isignore, gt_iscrowd)

    def match(self, affinity, match_thrs, gt_isignore=None, gt_iscrowd=None):
        raise NotImplementedError


@EVAL_MATCHERS.register_module()
class MatcherCoCo(BaseMatcher):

    def match(self, affinity, match_thrs, gt_isignore=None, gt_iscrowd=None):
        return match_coco(affinity, match_thrs, gt_isignore, gt_iscrowd)
