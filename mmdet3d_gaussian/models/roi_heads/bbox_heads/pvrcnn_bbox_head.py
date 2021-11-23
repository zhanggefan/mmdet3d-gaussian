# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn.bricks import build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet3d.core.bbox.structures import (LiDARInstance3DBoxes,
                                          rotation_3d_in_axis, xywhr2xyxyr)
from mmdet3d.models.builder import build_loss
from mmdet3d.ops import spconv as spconv
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class PVRCNNBboxHead(BaseModule):
    """PartA2 RoI head.

    Args:
        num_classes (int): The number of classes to prediction.
        seg_in_channels (int): Input channels of segmentation
            convolution layer.
        part_in_channels (int): Input channels of part convolution layer.
        seg_conv_channels (list(int)): Out channels of each
            segmentation convolution layer.
        part_conv_channels (list(int)): Out channels of each
            part convolution layer.
        merge_conv_channels (list(int)): Out channels of each
            feature merged convolution layer.
        down_conv_channels (list(int)): Out channels of each
            downsampled convolution layer.
        shared_fc_channels (list(int)): Out channels of each shared fc layer.
        cls_channels (list(int)): Out channels of each classification layer.
        reg_channels (list(int)): Out channels of each regression layer.
        dropout_ratio (float): Dropout ratio of classification and
            regression layers.
        roi_feat_size (int): The size of pooled roi features.
        with_corner_loss (bool): Whether to use corner loss or not.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for box head.
        conv_cfg (dict): Config dict of convolutional layers
        norm_cfg (dict): Config dict of normalization layers
        loss_bbox (dict): Config dict of box regression loss.
        loss_cls (dict): Config dict of classifacation loss.
    """

    def __init__(self,
                 in_channels,
                 grid_size,
                 num_classes,
                 class_agnostic=True,
                 shared_fc=(256, 256),
                 cls_fc=(256, 256),
                 reg_fc=(256, 256),
                 dropout=0.3,
                 with_corner_loss=True,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=1.0),
                 init_cfg=None):
        super(PVRCNNBboxHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.with_corner_loss = with_corner_loss
        self.class_agnostic = class_agnostic
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        cls_out_channels = 1 if class_agnostic else num_classes
        self.reg_out_channels = self.bbox_coder.code_size * cls_out_channels
        if self.use_sigmoid_cls:
            self.cls_out_channels = cls_out_channels
        else:
            self.cls_out_channels = cls_out_channels + 1

        self.dropout = dropout
        self.grid_size = grid_size
        self.norm_cfg = norm_cfg

        in_channels *= (self.grid_size ** 3)
        self.in_channels = in_channels

        self.shared_fc_layer = self.make_fc_layers(
            in_channels, shared_fc, range(len(shared_fc) - 1))
        self.cls_layers = self.make_fc_layers(
            shared_fc[-1], cls_fc, range(1))
        self.cls_out = nn.Conv1d(cls_fc[-1], self.cls_out_channels, 1,
                                 bias=True)
        self.reg_layers = self.make_fc_layers(
            shared_fc[-1], reg_fc, range(1))
        self.reg_out = nn.Conv1d(reg_fc[-1], self.reg_out_channels, 1,
                                 bias=True)

        if init_cfg is None:
            self.init_cfg = dict(
                type='Xavier',
                layer=['Conv2d', 'Conv1d'],
                distribution='uniform')

    def make_fc_layers(self, in_channels, fc, apply_dropout_indices):
        fc_layers = []
        pre_channel = in_channels
        for k, fck in enumerate(fc):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fck, 1, bias=False),
                build_norm_layer(self.norm_cfg, fck)[1],
                nn.ReLU()
            ])
            pre_channel = fck
            if self.dropout >= 0 and k in apply_dropout_indices:
                fc_layers.append(nn.Dropout(self.dropout))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def forward(self, feats):
        # (B * N, 6, 6, 6, C)
        rcnn_batch_size = feats.shape[0]
        feats = feats.permute(0, 4, 1, 2, 3).contiguous().view(
            rcnn_batch_size, -1, 1)
        # (BxN, C*6*6*6)
        shared_feats = self.shared_fc_layer(feats)
        cls_feats = self.cls_layers(shared_feats)
        reg_feats = self.reg_layers(shared_feats)

        cls_score = self.cls_out(cls_feats).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (B, 1)
        bbox_pred = self.reg_out(reg_feats).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (B, C)

        return cls_score, bbox_pred

    def loss(self, cls_score, bbox_pred, rois, labels, bbox_targets,
             pos_gt_bboxes, reg_mask, label_weights, bbox_weights):
        """Coumputing losses.

        Args:
            cls_score (torch.Tensor): Scores of each roi.
            bbox_pred (torch.Tensor): Predictions of bboxes.
            rois (torch.Tensor): Roi bboxes.
            labels (torch.Tensor): Labels of class.
            bbox_targets (torch.Tensor): Target of positive bboxes.
            pos_gt_bboxes (torch.Tensor): Ground truths of positive bboxes.
            reg_mask (torch.Tensor): Mask for positive bboxes.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_weights (torch.Tensor): Weights of bbox loss.

        Returns:
            dict: Computed losses.

                - loss_cls (torch.Tensor): Loss of classes.
                - loss_bbox (torch.Tensor): Loss of bboxes.
                - loss_corner (torch.Tensor): Loss of corners.
        """
        losses = dict()
        rcnn_batch_size = cls_score.shape[0]

        # calculate class loss
        cls_flat = cls_score.view(-1)
        loss_cls = self.loss_cls(cls_flat, labels, label_weights)
        losses['loss_cls'] = loss_cls

        # calculate regression loss
        code_size = self.bbox_coder.code_size
        pos_inds = (reg_mask > 0)
        if pos_inds.any() == 0:
            # fake a part loss
            losses['loss_bbox'] = loss_cls.new_tensor(0)
            if self.with_corner_loss:
                losses['loss_corner'] = loss_cls.new_tensor(0)
        else:
            pos_bbox_pred = bbox_pred.view(rcnn_batch_size, -1)[pos_inds]
            bbox_weights_flat = bbox_weights[pos_inds].view(-1, 1).repeat(
                1, pos_bbox_pred.shape[-1])
            loss_bbox = self.loss_bbox(
                pos_bbox_pred.unsqueeze(dim=0), bbox_targets.unsqueeze(dim=0),
                bbox_weights_flat.unsqueeze(dim=0))
            losses['loss_bbox'] = loss_bbox

            if self.with_corner_loss:
                pos_roi_boxes3d = rois[..., 1:].view(-1, code_size)[pos_inds]
                pos_roi_boxes3d = pos_roi_boxes3d.view(-1, code_size)
                batch_anchors = pos_roi_boxes3d.clone().detach()
                pos_rois_rotation = pos_roi_boxes3d[..., 6].view(-1)
                roi_xyz = pos_roi_boxes3d[..., 0:3].view(-1, 3)
                batch_anchors[..., 0:3] = 0
                # decode boxes
                pred_boxes3d = self.bbox_coder.decode(
                    batch_anchors,
                    pos_bbox_pred.view(-1, code_size)).view(-1, code_size)

                pred_boxes3d[..., 0:3] = rotation_3d_in_axis(
                    pred_boxes3d[..., 0:3].unsqueeze(1),
                    pos_rois_rotation,
                    axis=2).squeeze(1)

                pred_boxes3d[:, 0:3] += roi_xyz

                # calculate corner loss
                loss_corner = self.get_corner_loss_lidar(
                    pred_boxes3d, pos_gt_bboxes)
                losses['loss_corner'] = loss_corner.mean()

        return losses

    def get_targets(self, sampling_results, rcnn_train_cfg, concat=True):
        """Generate targets.

        Args:
            sampling_results (list[:obj:`SamplingResult`]):
                Sampled results from rois.
            rcnn_train_cfg (:obj:`ConfigDict`): Training config of rcnn.
            concat (bool): Whether to concatenate targets between batches.

        Returns:
            tuple[torch.Tensor]: Targets of boxes and class prediction.
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        iou_list = [res.iou for res in sampling_results]
        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            cfg=rcnn_train_cfg)

        (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
         bbox_weights) = targets

        if concat:
            label = torch.cat(label, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            pos_gt_bboxes = torch.cat(pos_gt_bboxes, 0)
            reg_mask = torch.cat(reg_mask, 0)

            label_weights = torch.cat(label_weights, 0)
            label_weights /= torch.clamp(label_weights.sum(), min=1.0)

            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_weights /= torch.clamp(bbox_weights.sum(), min=1.0)

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def _get_target_single(self, pos_bboxes, pos_gt_bboxes, ious, cfg):
        """Generate training targets for a single sample.

        Args:
            pos_bboxes (torch.Tensor): Positive boxes with shape
                (N, 7).
            pos_gt_bboxes (torch.Tensor): Ground truth boxes with shape
                (M, 7).
            ious (torch.Tensor): IoU between `pos_bboxes` and `pos_gt_bboxes`
                in shape (N, M).
            cfg (dict): Training configs.

        Returns:
            tuple[torch.Tensor]: Target for positive boxes.
                (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)
        """
        cls_pos_mask = ious > cfg.cls_pos_thr
        cls_neg_mask = ious < cfg.cls_neg_thr
        interval_mask = (cls_pos_mask == 0) & (cls_neg_mask == 0)

        # iou regression target
        label = (cls_pos_mask > 0).float()
        label[interval_mask] = ious[interval_mask] * 2 - 0.5
        # label weights
        label_weights = (label >= 0).float()

        # box regression target
        reg_mask = pos_bboxes.new_zeros(ious.size(0)).long()
        reg_mask[0:pos_gt_bboxes.size(0)] = 1
        bbox_weights = (reg_mask > 0).float()
        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach()
            roi_center = pos_bboxes[..., 0:3]
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi)

            # canonical transformation
            pos_gt_bboxes_ct[..., 0:3] -= roi_center
            pos_gt_bboxes_ct[..., 6] -= roi_ry
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1), -roi_ry,
                axis=2).squeeze(1)

            # flip orientation if rois have opposite orientation
            ry_label = pos_gt_bboxes_ct[..., 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (
                    2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = ry_label > np.pi
            ry_label[flag] = ry_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
            pos_gt_bboxes_ct[..., 6] = ry_label

            rois_anchor = pos_bboxes.clone().detach()
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            bbox_targets = self.bbox_coder.encode(rois_anchor,
                                                  pos_gt_bboxes_ct)
        else:
            # no fg bbox
            bbox_targets = pos_gt_bboxes.new_empty((0, 7))

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def get_corner_loss_lidar(self, pred_bbox3d, gt_bbox3d, delta=1):
        """Calculate corner loss of given boxes.

        Args:
            pred_bbox3d (torch.FloatTensor): Predicted boxes in shape (N, 7).
            gt_bbox3d (torch.FloatTensor): Ground truth boxes in shape (N, 7).

        Returns:
            torch.FloatTensor: Calculated corner loss in shape (N).
        """
        assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

        # This is a little bit hack here because we assume the box for
        # Part-A2 is in LiDAR coordinates
        gt_boxes_structure = LiDARInstance3DBoxes(gt_bbox3d)
        pred_box_corners = LiDARInstance3DBoxes(pred_bbox3d).corners
        gt_box_corners = gt_boxes_structure.corners

        # This flip only changes the heading direction of GT boxes
        gt_bbox3d_flip = gt_boxes_structure.clone()
        gt_bbox3d_flip.tensor[:, 6] += np.pi
        gt_box_corners_flip = gt_bbox3d_flip.corners

        corner_dist = torch.min(
            torch.norm(pred_box_corners - gt_box_corners, dim=2),
            torch.norm(pred_box_corners - gt_box_corners_flip,
                       dim=2))  # (N, 8)
        # huber loss
        abs_error = torch.abs(corner_dist)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = (abs_error - quadratic)
        corner_loss = 0.5 * quadratic ** 2 + delta * linear

        return corner_loss.mean(dim=1)

    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   class_labels,
                   class_pred,
                   img_metas,
                   cfg=None):
        """Generate bboxes from bbox head predictions.

        Args:
            rois (torch.Tensor): Roi bounding boxes.
            cls_score (torch.Tensor): Scores of bounding boxes.
            bbox_pred (torch.Tensor): Bounding boxes predictions
            class_labels (torch.Tensor): Label of classes
            class_pred (torch.Tensor): Score for nms.
            img_metas (list[dict]): Point cloud and image's meta info.
            cfg (:obj:`ConfigDict`): Testing config.

        Returns:
            list[tuple]: Decoded bbox, scores and labels after nms.
        """
        roi_batch_id = rois[..., 0]
        roi_boxes = rois[..., 1:]  # boxes without batch id
        batch_size = int(roi_batch_id.max().item() + 1)

        # decode boxes
        roi_ry = roi_boxes[..., 6].view(-1)
        roi_xyz = roi_boxes[..., 0:3].view(-1, 3)
        local_roi_boxes = roi_boxes.clone().detach()
        local_roi_boxes[..., 0:3] = 0
        rcnn_boxes3d = self.bbox_coder.decode(local_roi_boxes, bbox_pred)
        rcnn_boxes3d[..., 0:3] = rotation_3d_in_axis(
            rcnn_boxes3d[..., 0:3].unsqueeze(1), roi_ry, axis=2).squeeze(1)
        rcnn_boxes3d[:, 0:3] += roi_xyz

        # post processing
        result_list = []
        for batch_id in range(batch_size):
            cur_class_labels = class_labels[batch_id]
            cur_cls_score = cls_score[roi_batch_id == batch_id].view(-1)

            cur_box_prob = class_pred[batch_id]
            cur_rcnn_boxes3d = rcnn_boxes3d[roi_batch_id == batch_id]
            selected = self.multi_class_nms(cur_box_prob, cur_rcnn_boxes3d,
                                            cfg.score_thr, cfg.nms_thr,
                                            img_metas[batch_id],
                                            cfg.use_rotate_nms)
            selected_bboxes = cur_rcnn_boxes3d[selected]
            selected_label_preds = cur_class_labels[selected]
            selected_scores = cur_cls_score[selected]

            result_list.append(
                (img_metas[batch_id]['box_type_3d'](selected_bboxes,
                                                    self.bbox_coder.code_size),
                 selected_scores, selected_label_preds))
        return result_list

    def multi_class_nms(self,
                        box_probs,
                        box_preds,
                        score_thr,
                        nms_thr,
                        input_meta,
                        use_rotate_nms=True):
        """Multi-class NMS for box head.

        Note:
            This function has large overlap with the `box3d_multiclass_nms`
            implemented in `mmdet3d.core.post_processing`. We are considering
            merging these two functions in the future.

        Args:
            box_probs (torch.Tensor): Predicted boxes probabitilies in
                shape (N,).
            box_preds (torch.Tensor): Predicted boxes in shape (N, 7+C).
            score_thr (float): Threshold of scores.
            nms_thr (float): Threshold for NMS.
            input_meta (dict): Meta informations of the current sample.
            use_rotate_nms (bool, optional): Whether to use rotated nms.
                Defaults to True.

        Returns:
            torch.Tensor: Selected indices.
        """
        if use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu

        assert box_probs.shape[
                   1] == self.num_classes, f'box_probs shape: {str(box_probs.shape)}'
        selected_list = []
        selected_labels = []
        boxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            box_preds, self.bbox_coder.code_size).bev)

        score_thresh = score_thr if isinstance(
            score_thr, list) else [score_thr for x in range(self.num_classes)]
        nms_thresh = nms_thr if isinstance(
            nms_thr, list) else [nms_thr for x in range(self.num_classes)]
        for k in range(0, self.num_classes):
            class_scores_keep = box_probs[:, k] >= score_thresh[k]

            if class_scores_keep.int().sum() > 0:
                original_idxs = class_scores_keep.nonzero(
                    as_tuple=False).view(-1)
                cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
                cur_rank_scores = box_probs[class_scores_keep, k]

                cur_selected = nms_func(cur_boxes_for_nms, cur_rank_scores,
                                        nms_thresh[k])

                if cur_selected.shape[0] == 0:
                    continue
                selected_list.append(original_idxs[cur_selected])
                selected_labels.append(
                    torch.full([cur_selected.shape[0]],
                               k + 1,
                               dtype=torch.int64,
                               device=box_preds.device))

        selected = torch.cat(
            selected_list, dim=0) if len(selected_list) > 0 else []
        return selected
