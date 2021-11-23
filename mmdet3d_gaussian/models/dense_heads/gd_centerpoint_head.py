import torch
from mmcv.runner import force_fp32, BaseModule

from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.core import (
    circle_nms, draw_heatmap_gaussian, gaussian_radius, xywhr2xyxyr)
from mmdet3d.models.dense_heads import CenterHead
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import multi_apply
from mmcv.cnn import CONV_LAYERS


@CONV_LAYERS.register_module('ConvDS')
class ConvDS(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', init_cfg=None):
        if init_cfg is None:
            init_cfg = dict(type='kaiming_init', layer='Conv2d',
                            override=dict(type='kaiming_init', name='chn_conv',
                                          nonlinearity='Linear'))
        super(ConvDS, self).__init__(init_cfg)
        self.chn_conv = torch.nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=in_channels,
                                        bias=False,
                                        padding_mode=padding_mode)
        self.dep_conv = torch.nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=bias)
        self.in_channels = self.chn_conv.in_channels
        self.out_channels = self.dep_conv.out_channels
        self.kernel_size = self.chn_conv.kernel_size
        self.stride = self.chn_conv.stride
        self.padding = self.chn_conv.padding
        self.dilation = self.chn_conv.dilation
        self.transposed = self.chn_conv.transposed
        self.output_padding = self.dep_conv.output_padding
        self.groups = self.chn_conv.groups
        self.padding_mode = self.chn_conv.padding_mode

    def forward(self, input):
        return self.dep_conv(self.chn_conv(input))


@HEADS.register_module(name='CenterHead', force=True)
class CenterHeadRev(CenterHead):

    def _gather_feat(self, feat, ind):
        batch_ind = ind[..., 0]
        x_ind = ind[..., 1]
        y_ind = ind[..., 2]
        return feat[batch_ind, :, y_ind, x_ind]

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        heatmaps, anno_boxes, pos_inds = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        heatmaps = [torch.stack(heatmap) for heatmap in zip(*heatmaps)]
        # transpose anno_boxes
        anno_boxes = [torch.cat(anno_box, dim=0) for anno_box in
                      zip(*anno_boxes)]
        # transpose inds
        batch_pos_inds = []
        for pos_ind in zip(*pos_inds):
            batch_prefix = [ind.new_full((ind.size(0), 1), b) for
                            b, ind in enumerate(pos_ind)]
            batch_prefix = torch.cat(batch_prefix, dim=0)
            pos_ind = torch.cat(pos_ind, dim=0)
            batch_pos_inds.append(
                torch.cat((batch_prefix, pos_ind), dim=-1))
        return heatmaps, anno_boxes, batch_pos_inds

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, pos_inds = [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[0],
                 feature_map_size[1]))

            out_size_factor = self.train_cfg['out_size_factor']

            width = task_boxes[idx][:, 3] / voxel_size[0] / out_size_factor
            length = task_boxes[idx][:, 4] / voxel_size[1] / out_size_factor
            x_ind = ((task_boxes[idx][:, 0] - pc_range[0]) / voxel_size[
                0] / out_size_factor).long()
            y_ind = ((task_boxes[idx][:, 1] - pc_range[1]) / voxel_size[
                1] / out_size_factor).long()

            valid = width.gt(0) * length.gt(0)
            valid *= x_ind.ge(0) * x_ind.lt(feature_map_size[1])
            valid *= y_ind.ge(0) * y_ind.lt(feature_map_size[0])

            center_xy_int = torch.stack((x_ind, y_ind), dim=-1)
            width_cpu = width.cpu()
            length_cpu = length.cpu()

            for k in valid.nonzero(as_tuple=True)[0].cpu():
                cls_id = task_classes[idx][k] - 1

                radius = gaussian_radius((length_cpu[k], width_cpu[k]),
                                         min_overlap=self.train_cfg[
                                             'gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))

                draw_gaussian(heatmap[cls_id], center_xy_int[k], radius)

            heatmaps.append(heatmap)
            anno_boxes.append(task_boxes[idx][valid])
            pos_inds.append(center_xy_int[valid])
        return heatmaps, anno_boxes, pos_inds

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, pos_inds = self.get_targets(gt_bboxes_3d,
                                                          gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = self.bbox_coder.encode(anno_boxes[task_id])

            # Regression loss for dimension, offset, height, rotation
            pos_ind = pos_inds[task_id]
            pred = self._reconstruct_bbox(preds_dict[0])
            pred = self._gather_feat(pred, pos_ind)

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = target_box.new_tensor(code_weights).unsqueeze(
                0).expand_as(target_box)

            if target_box.numel():
                loss_bbox = self.loss_bbox(pred, target_box, bbox_weights,
                                           avg_factor=max(num_pos, 1))
            else:
                loss_bbox = loss_heatmap.new_zeros((1,))

            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
        return loss_dict

    def _reconstruct_bbox(self, preds_dict):
        pred = []
        if 'reg' in preds_dict:
            pred.append(preds_dict['reg'])
        else:
            batch, _, h, w = preds_dict['height'].shape
            pred.append(preds_dict['height'].new_full((batch, 2, h, w), 0.5))
        pred.append(preds_dict['height'])
        pred.append(preds_dict['dim'])
        pred.append(preds_dict['rot'])

        if 'vel' in preds_dict:
            pred.append(preds_dict['vel'])

        return torch.cat(pred, dim=1)

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        post_center_limit_range = self.test_cfg.get('post_center_limit_range',
                                                    None)
        max_per_img = self.test_cfg.get('max_per_img', 128)
        score_threshold = self.test_cfg.get('score_threshold', 0.1)

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_pred = self._reconstruct_bbox(preds_dict[0])

            scores, clses, locs, preds = self.bbox_coder.select_best(
                batch_heatmap, batch_pred, max_per_img)

            preds = self.bbox_coder.decode(locs, preds)

            mask = scores.ge(score_threshold)
            if post_center_limit_range is not None:
                for i in range(3):
                    mask *= preds[..., i].ge(post_center_limit_range[i]).le(
                        post_center_limit_range[i + 3])

            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_bboxes = [preds[i][mask[i]] for i in range(batch_size)]
            batch_scores = [scores[i][mask[i]] for i in range(batch_size)]
            batch_labels = [clses[i][mask[i]] for i in range(batch_size)]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = batch_bboxes[i]
                    scores = batch_scores[i]
                    labels = batch_labels[i]
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(batch_scores, batch_bboxes,
                                             batch_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, batch_scores, batch_bboxes, batch_labels,
                            img_metas):
        """Rotate nms for each task.

        Args:
            batch_scores (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_bboxes (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the \
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the \
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the \
                    shape of [N].
        """
        predictions_dicts = []
        for i, (bboxes, scores, labels) in enumerate(
                zip(batch_bboxes, batch_scores, batch_labels)):

            # Apply NMS in birdeye view
            # get highest score per prediction, then apply nms
            # to remove overlapped box.
            if scores.numel() > 0:
                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    bboxes[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_gpu(
                    boxes_for_nms,
                    scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = bboxes[selected]
            selected_labels = labels[selected]
            selected_scores = scores[selected]

            # finally generate predictions.
            predictions_dict = dict(
                bboxes=selected_boxes,
                scores=selected_scores,
                labels=selected_labels)

            predictions_dicts.append(predictions_dict)
        return predictions_dicts


@HEADS.register_module()
class CenterGDHead(CenterHeadRev):
    def __init__(self,
                 loss_gd,
                 *args, **kwargs):
        super(CenterGDHead, self).__init__(*args, **kwargs)
        self.loss_gd = build_loss(loss_gd)

    def _reconstruct_bbox(self, preds_dict):
        pred = []
        if 'reg' in preds_dict:
            pred.append(preds_dict['reg'])
        else:
            batch, _, h, w = preds_dict['height'].shape
            pred.append(preds_dict['height'].new_full((batch, 2, h, w), 0.5))
        pred.append(preds_dict['height'])
        pred.append(preds_dict['dim'])
        pred.append(preds_dict['yaw'])
        pred.append(preds_dict['dir'])

        if 'vel' in preds_dict:
            pred.append(preds_dict['vel'])

        return torch.cat(pred, dim=1)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        heatmaps, anno_boxes, pos_inds = self.get_targets(gt_bboxes_3d,
                                                          gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))

            target_box = self.bbox_coder.encode(anno_boxes[task_id])
            target_l1 = target_box[..., 7:]
            target_gd = target_box[..., :7]

            # Regression loss for dimension, offset, height, rotation
            pos_ind = pos_inds[task_id]
            pred = self._reconstruct_bbox(preds_dict[0])
            pred = self._gather_feat(pred, pos_ind)

            pred_gd = self.bbox_coder.decode(pos_ind[..., 1:], pred,
                                             correct_yaw=False)[..., :7]
            pred_l1 = pred[..., 7:]

            weights_l1 = self.train_cfg.get('code_weights', None)
            weights_l1 = target_l1.new_tensor(weights_l1).unsqueeze(0)
            weights_l1 = weights_l1.expand_as(target_l1)

            if target_box.numel():
                loss_l1 = self.loss_bbox(pred_l1, target_l1, weights_l1,
                                         avg_factor=max(num_pos, 1))
                loss_gd = self.loss_gd(pred_gd, target_gd,
                                       avg_factor=max(num_pos, 1))
            else:
                loss_l1 = loss_heatmap.new_zeros((1,))
                loss_gd = loss_heatmap.new_zeros((1,))

            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_l1'] = loss_l1
            loss_dict[f'task{task_id}.loss_gd'] = loss_gd

        return loss_dict
