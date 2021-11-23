from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.transforms_3d import ObjectSample
import numpy as np


@PIPELINES.register_module()
class NormalizeIntensityTanh:
    def __init__(self, intensity_column=3, pre_tanh_gain=1.,
                 post_tanh_gain=1.):
        self.intensity_column = intensity_column
        self.pre_tanh_gain = pre_tanh_gain
        self.post_tanh_gain = post_tanh_gain

    def __call__(self, input_dict):
        intensity = input_dict['points'].tensor[:, self.intensity_column]
        intensity = self.post_tanh_gain * (
                self.pre_tanh_gain * intensity).tanh()
        input_dict['points'].tensor[:, self.intensity_column] = intensity
        input_dict['intensity_norm_cfg'] = dict(
            pre_tanh_gain=self.pre_tanh_gain,
            post_tanh_gain=self.post_tanh_gain)
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pre_tanh_gain={self.pre_tanh_gain}, '
        repr_str += f'post_tanh_gain={self.post_tanh_gain})'
        return repr_str


@PIPELINES.register_module()
class LabelIDMap:
    def __init__(self, map):
        self.map = map
        olds, news = [], []
        for old, new in map:
            assert old not in olds
            olds.append(old)
            if new not in news:
                news.append(new)
        id_map = []
        for idx, (old, new) in enumerate(self.map):
            id_map.append((idx, news.index(new)))
        self.id_map = id_map

    def __call__(self, input_dict):
        for old, new in self.id_map:
            input_dict['gt_labels_3d'][input_dict['gt_labels_3d'] == old] = new

        if 'gt_names_3d' in input_dict:
            for old, new in self.map:
                input_dict['gt_names_3d'][
                    input_dict['gt_names_3d'] == old] = new

        if 'gt_names' in input_dict:
            for old, new in self.map:
                input_dict['gt_names'][input_dict['gt_names'] == old] = new

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(map={self.map})'
        return repr_str


@PIPELINES.register_module(name='ObjectSample', force=True)
class ObjectSampleRev(ObjectSample):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, use_ground_plane=False):
        super(ObjectSampleRev, self).__init__(db_sampler, sample_2d)
        self.use_ground_plane = use_ground_plane

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation,
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        if self.use_ground_plane and 'plane' in input_dict['ann_info']:
            ground_plane = input_dict['ann_info']['plane']
            input_dict['plane'] = ground_plane
        else:
            ground_plane = None
        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                img=None,
                ground_plane=ground_plane)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str
