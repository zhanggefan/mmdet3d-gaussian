import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmdet3d.core import Box3DMode, Coord3DMode
from ...core import show_result


class ShowResultMixin(object):
    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            if 'gt_bboxes_3d' in data:
                if isinstance(data['gt_bboxes_3d'][0], DC):
                    gt_bboxes = data['gt_bboxes_3d'][0]._data[0][batch_id]
                elif mmcv.is_list_of(data['gt_bboxes_3d'][0], dict):
                    gt_bboxes = data['gt_bboxes_3d'][0][batch_id]
                else:
                    ValueError(
                        "Unsupported data type "
                        f"{type(data['gt_bboxes_3d'][0])} "
                        f'for visualization!')
                gt_bboxes = Box3DMode.convert(gt_bboxes, box_mode_3d,
                                              Box3DMode.DEPTH)
                gt_bboxes = gt_bboxes.tensor.cpu().numpy()
            else:
                gt_bboxes = None

            file_name = osp.split(pts_filename)[-1].split('.')[0]

            pred_bboxes = result[batch_id]['boxes_3d']

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for convertion!')
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name)
