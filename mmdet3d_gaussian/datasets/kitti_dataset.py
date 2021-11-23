from mmdet.datasets import DATASETS
from mmdet3d.datasets import KittiDataset
from mmdet3d.core.bbox import Box3DMode
from ..core.evaluation import eval_map_flexible
import numpy as np


@DATASETS.register_module(name='KittiDataset', force=True)
class KittiDatasetRev(KittiDataset):
    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if metric == ['cowa'] or metric in ['cowa']:
            gt_annos = []
            for i in range(len(self)):
                gt_i = self.get_ann_info(i)
                bboxes = gt_i['gt_bboxes_3d'].convert_to(
                    Box3DMode.LIDAR).tensor.numpy()
                labels = gt_i['gt_labels_3d']
                gt_annos.append(dict(
                    gt_bboxes=bboxes,
                    gt_labels=labels,
                    gt_attrs={'ignore': labels == -1}
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
                    [bboxes[labels == cls] for cls in
                     range(len(self.CLASSES))])

            return eval_map_flexible(
                det_results, gt_annos, match_thrs=[0.5, 0.7],
                breakdowns=[],
                matcher=dict(type='MatcherCoCo'),
                classes=self.CLASSES, logger=logger,
                report_config=[
                    ('car_70',
                     lambda x: x['class_name'] == 'Car' and
                               x['match_threshold'] == 0.7 and
                               x['breakdown'] == 'All'),
                    ('ped_50',
                     lambda x: x['class_name'] == 'Pedestrian' and
                               x['match_threshold'] == 0.5 and
                               x['breakdown'] == 'All'),
                    ('cyc_50',
                     lambda x: x['class_name'] == 'Cyclist' and
                               x['match_threshold'] == 0.5 and
                               x['breakdown'] == 'All')
                ], nproc=None)
        else:
            return super(KittiDatasetRev, self).evaluate(
                results, metric, logger, pklfile_prefix, submission_prefix,
                show, out_dir, pipeline)

    def get_data_info(self, index):
        input_dict = super(KittiDatasetRev, self).get_data_info(index)

        if self.test_mode and 'annos' in self.data_infos[index]:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        # convert gt_bboxes_3d to velodyne coordinates
        reverse = np.linalg.inv(rect @ Trv2c)

        (plane_n_c, plane_pt_c) = (info['plane'][:3],
                                   -info['plane'][:3] * info['plane'][3])
        plane_n_l = (reverse[:3, :3] @ plane_n_c[:, None])[:, 0]
        plane_pt_l = (
                reverse[:3, :3] @ plane_pt_c[:, None][:, 0] + reverse[:3, 3])
        plane_l = np.zeros_like(plane_n_l, shape=(4,))
        plane_l[:3] = plane_n_l
        plane_l[3] = -plane_n_l.T @ plane_pt_l

        anns_results = super(KittiDatasetRev, self).get_ann_info(index)

        anns_results['plane'] = plane_l
        anns_results['difficulty'] = info['annos']['difficulty']

        return anns_results
