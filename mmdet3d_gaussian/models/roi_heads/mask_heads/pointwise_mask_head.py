# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import build_norm_layer
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.models.builder import build_loss
from mmdet.core import multi_apply
from mmdet.models import HEADS


@HEADS.register_module()
class PointwiseMaskHead(BaseModule):
    def __init__(self,
                 in_channels,
                 num_classes=3,
                 mlps=(256, 256),
                 extra_width=0.2,
                 class_agnostic=False,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg=None,
                 loss_seg=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0)):
        super(PointwiseMaskHead, self).__init__(init_cfg=init_cfg)
        self.extra_width = extra_width
        self.class_agnostic = class_agnostic
        self.num_classes = num_classes

        self.in_channels = in_channels
        self.use_sigmoid_cls = loss_seg.get('use_sigmoid', False)

        out_channels = 1 if class_agnostic else num_classes
        if self.use_sigmoid_cls:
            self.out_channels = out_channels
        else:
            self.out_channels = out_channels + 1

        mlps_layers = []
        cin = in_channels
        for cout in mlps:
            mlps_layers.extend([
                nn.Linear(cin, cout, bias=False),
                build_norm_layer(norm_cfg, cout)[1],
                nn.ReLU()])
            cin = cout
        mlps_layers.append(nn.Linear(cin, self.out_channels, bias=True))

        self.seg_cls_layer = nn.Sequential(*mlps_layers)

        self.loss_seg = build_loss(loss_seg)

    def forward(self, feats):
        seg_preds = self.seg_cls_layer(feats)  # (N, 1)
        return dict(seg_preds=seg_preds)

    def get_targets_single(self, point_xyz, gt_bboxes_3d, gt_labels_3d):
        """generate segmentation and part prediction targets for a single
        sample.

        Args:
            voxel_centers (torch.Tensor): The center of voxels in shape
                (voxel_num, 3).
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes in
                shape (box_num, 7).
            gt_labels_3d (torch.Tensor): Class labels of ground truths in
                shape (box_num).

        Returns:
            tuple[torch.Tensor]: Segmentation targets with shape [voxel_num]
                part prediction targets with shape [voxel_num, 3]
        """
        gt_bboxes_3d = gt_bboxes_3d.to(point_xyz.device)
        enlarged_gt_boxes = gt_bboxes_3d.enlarged_box(self.extra_width)

        box_idx = gt_bboxes_3d.points_in_boxes_part(point_xyz).long()
        enlarge_box_idx = enlarged_gt_boxes.points_in_boxes_part(
            point_xyz).long()

        gt_labels_pad = F.pad(
            gt_labels_3d, (1, 0), mode='constant', value=self.num_classes)
        seg_targets = gt_labels_pad[(box_idx + 1)]
        fg_pt_flag = box_idx > -1
        ignore_flag = fg_pt_flag ^ (enlarge_box_idx > -1)
        seg_targets[ignore_flag] = -1

        return seg_targets,

    def get_targets(self, points_bxyz, gt_bboxes_3d, gt_labels_3d):
        """generate segmentation and part prediction targets.

        Args:
            xyz (torch.Tensor): The center of voxels in shape
                (B, num_points, 3).
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes in
                shape (box_num, 7).
            gt_labels_3d (torch.Tensor): Class labels of ground truths in
                shape (box_num).

        Returns:
            dict: Prediction targets

                - seg_targets (torch.Tensor): Segmentation targets
                    with shape [voxel_num].
                - part_targets (torch.Tensor): Part prediction targets
                    with shape [voxel_num, 3].
        """
        batch_size = len(gt_labels_3d)
        points_xyz_list = []
        for idx in range(batch_size):
            coords_idx = points_bxyz[:, 0] == idx
            points_xyz_list.append(points_bxyz[coords_idx][..., 1:])
        seg_targets, = multi_apply(self.get_targets_single,
                                   points_xyz_list, gt_bboxes_3d,
                                   gt_labels_3d)
        seg_targets = torch.cat(seg_targets, dim=0)
        return dict(seg_targets=seg_targets)

    def loss(self, semantic_results, semantic_targets):
        seg_preds = semantic_results['seg_preds']
        seg_targets = semantic_targets['seg_targets']

        pos_mask = (seg_targets > -1) & (seg_targets < self.num_classes)

        pos = pos_mask.float()
        neg = (seg_targets == self.num_classes).float()
        seg_weights = pos + neg
        pos_normalizer = pos.sum()
        seg_weights = seg_weights / torch.clamp(pos_normalizer, min=1.0)

        if self.class_agnostic:
            seg_cls_target = pos_mask.long()
        else:
            seg_cls_target = seg_targets.masked_fill(
                seg_targets < 0, self.num_classes)

        loss_seg = self.loss_seg(seg_preds, seg_cls_target, seg_weights)

        return dict(loss_seg=loss_seg)
