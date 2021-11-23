import numpy as np
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
from ..core.evaluation import eval_map_flexible


@DATASETS.register_module()
class CowaDataset(KittiDataset):
    CLASSES = ('big_vehicle',
               'motorcycle',
               'pedestrian',
               'vehicle',
               'bicycle',
               'huge_vehicle',
               'cone',
               'tricycle')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='m1',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=(0, -61.44, -2, 122.88, 61.44, 4)):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:07d}.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        sample_idx = info['sample_idx']

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename)

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        gt_bboxes_3d = annos['bboxes'].astype(np.float32)
        gt_names = annos['name']

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d)

        gt_labels_3d = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d).astype(np.int64)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names)
        return anns_results

    def evaluate(self,
                 results,
                 metric='cowa',
                 class_names_mapping=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        accept_type = ['cowa']
        if isinstance(metric, list):
            for m in metric:
                assert m in accept_type, f'invalid metric {m}'
        else:
            assert metric in accept_type, f'invalid metric {metric}'

        id_map = None
        eval_classes = self.CLASSES
        if class_names_mapping is not None:
            olds, eval_classes = [], []
            for old, new in class_names_mapping:
                assert old not in olds
                olds.append(old)
                if new not in eval_classes:
                    eval_classes.append(new)
            id_map = []
            for idx, (old, new) in enumerate(class_names_mapping):
                id_map.append((idx, eval_classes.index(new)))

        gt_annos = []
        xy_reange = [self.pcd_limit_range[i] for i in [0, 1, 3, 4]]
        for i in range(len(self)):
            gt_i = self.get_ann_info(i)
            bboxes = gt_i['gt_bboxes_3d'].convert_to(
                Box3DMode.LIDAR).tensor.numpy()
            labels = gt_i['gt_labels_3d']
            if id_map is not None:
                for old, new in id_map:
                    labels[labels == old] = new
            in_range = gt_i['gt_bboxes_3d'].in_range_bev(xy_reange).numpy()
            gt_annos.append(dict(
                gt_bboxes=bboxes,
                gt_labels=labels,
                gt_attrs={'ignore': (~in_range)}
            ))
        det_results = []
        for i in range(len(results)):
            res_i = results[i]
            if 'pts_bbox' in results[i]:
                res_i = results[i]['pts_bbox']
            bboxes = res_i['boxes_3d'].convert_to(
                Box3DMode.LIDAR).tensor.numpy()
            labels = res_i['labels_3d'].numpy()
            scores = res_i['scores_3d'].numpy()
            bboxes = np.concatenate((bboxes, scores[:, None]), axis=-1)
            det_results.append(
                [bboxes[labels == cls] for cls in range(len(eval_classes))])

        return eval_map_flexible(
            det_results, gt_annos, match_thrs=[0.5, 0.7],
            breakdowns=[
                dict(
                    type='RangeBreakdown',
                    ranges=dict(
                        Dist_Near=(0, 30),
                        Dist_Middle=(30, 50),
                        Dist_Far=(50, 10000)))],
            matcher=dict(type='MatcherCoCo'),
            classes=eval_classes, logger=logger,
            report_config=[
                ('car_70',
                 lambda x: x['class_name'] == 'vehicle' and
                           x['match_threshold'] == 0.7 and
                           x['breakdown'] == 'All'),
                ('bcar_70',
                 lambda x: x['class_name'] == 'big_vehicle' and
                           x['match_threshold'] == 0.7 and
                           x['breakdown'] == 'All'),
                ('hcar_70',
                 lambda x: x['class_name'] == 'huge_vehicle' and
                           x['match_threshold'] == 0.7 and
                           x['breakdown'] == 'All'),
                ('ped_50',
                 lambda x: x['class_name'] == 'pedestrian' and
                           x['match_threshold'] == 0.5 and
                           x['breakdown'] == 'All'),
                ('cyc_50',
                 lambda x: x['class_name'] == 'motorcycle_bicycle' and
                           x['match_threshold'] == 0.5 and
                           x['breakdown'] == 'All'),
                ('tri_50',
                 lambda x: x['class_name'] == 'tricycle' and
                           x['match_threshold'] == 0.5 and
                           x['breakdown'] == 'All'),
                ('cone_50',
                 lambda x: x['class_name'] == 'cone' and
                           x['match_threshold'] == 0.5 and
                           x['breakdown'] == 'All'),
            ], nproc=None)
