import numpy as np
from .builder import EVAL_BREAKDOWNS


@EVAL_BREAKDOWNS.register_module()
class NoBreakdown:

    def __init__(self, classes, apply_to=None, *args, **kwargs):
        if apply_to is None:
            apply_to = classes
        self.classes = classes
        self.apply_to = apply_to
        self.names = ['All']

    def breakdown_flags(self, boxes, attrs=None):
        num_boxes = len(boxes)
        flags = np.ones((1, num_boxes), dtype=np.bool)
        if attrs is not None and 'ignore' in attrs:
            flags[:, attrs['ignore']] = False
        return flags

    def breakdown(self, boxes, label, attrs=None):
        flags = self.breakdown_flags(boxes, attrs)
        if self.classes[label] in self.apply_to:
            return flags
        else:
            return flags[:0]

    def breakdown_names(self, label):
        if self.classes[label] in self.apply_to:
            return [f'{n}' for n in self.names]
        else:
            return []


@EVAL_BREAKDOWNS.register_module()
class RangeBreakdown(NoBreakdown):

    def __init__(self, ranges, classes, apply_to=None, *args, **kwargs):
        super(RangeBreakdown, self).__init__(classes, apply_to, *args,
                                             **kwargs)
        self.names = []
        self.ranges = []
        for k in ranges:
            self.names.append(k)
            self.ranges.append(ranges[k])

    def breakdown_flags(self, boxes, attrs=None):
        num_ranges = len(self.ranges)
        num_boxes = len(boxes)
        if attrs is not None and 'distance' in attrs:
            distance = attrs['distance']
        else:
            xyz = boxes[:, :3]
            distance = np.linalg.norm(xyz, axis=-1)
        dist_flags = np.zeros((num_ranges, num_boxes), dtype=np.bool)
        for range_idx, (min_dist, max_dist) in enumerate(self.ranges):
            dist_flags[range_idx][
                (distance >= min_dist) & (distance < max_dist)] = True
        if attrs is not None and 'ignore' in attrs:
            dist_flags[:, attrs['ignore']] = False
        return dist_flags


@EVAL_BREAKDOWNS.register_module()
class VolumeBreakdown(NoBreakdown):
    def __init__(self, ranges, classes, apply_to=None, *args, **kwargs):
        super(VolumeBreakdown, self).__init__(classes, apply_to, *args,
                                              **kwargs)
        self.names = []
        self.ranges = []
        for k in ranges:
            self.names.append(k)
            self.ranges.append(ranges[k])

    def breakdown_flags(self, boxes, attrs=None):
        num_ranges = len(self.ranges)
        num_boxes = len(boxes)
        if attrs is not None and 'volumn' in attrs:
            vol = attrs['volumn']
        else:
            vol = np.prod(boxes[:, 3:6], axis=-1)
        vol_flags = np.zeros((num_ranges, num_boxes), dtype=np.bool)
        for vol_idx, (min_vol, max_vol) in enumerate(self.ranges):
            vol_flags[vol_idx, (vol >= min_vol) & (vol < max_vol)] = True
        if attrs is not None and 'ignore' in attrs:
            vol_flags[:, attrs['ignore']] = False
        return vol_flags
