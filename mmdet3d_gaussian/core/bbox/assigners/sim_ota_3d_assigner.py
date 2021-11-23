from mmdet.core.bbox.assigners import BaseAssigner, AssignResult
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet3d.ops import points_in_boxes_all
import torch
import numpy as np
import torch.nn.functional as F
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes


@BBOX_ASSIGNERS.register_module()
class SimOTABEVAssigner(BaseAssigner):
    INF = 100000000
    EPS = 0.00000001
    """
    used in combination with MlvlPointGenerator
    """

    def __init__(self,
                 center_radius=0.5,
                 candidate_topk=10,
                 iou_weight=3.0,
                 dir_weight=0.0,
                 cls_weight=1.0,
                 match_init=2.0,
                 debug=False):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.dir_weight = dir_weight
        self.cls_weight = cls_weight
        self.match_init = match_init
        self.debug = debug

    def assign(self,
               pred_scores,
               decoded_bboxes,
               priors,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to priors using SimOTA.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = priors.new_full((num_bboxes,),
                                           0,
                                           dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1,
                                                          dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        valid_decoded_bbox = LiDARInstance3DBoxes(valid_decoded_bbox)
        gt_bboxes = LiDARInstance3DBoxes(gt_bboxes)

        pairwise_ious = LiDARInstance3DBoxes.overlaps(valid_decoded_bbox,
                                                      gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + self.EPS)

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        cls_cost = F.binary_cross_entropy(
            valid_pred_scores.sqrt_(), gt_onehot_label,
            reduction='none').sum(-1)

        cost_matrix = cls_cost * self.cls_weight + iou_cost * self.iou_weight
        cost_matrix[is_in_boxes_and_center] = cost_matrix[
            is_in_boxes_and_center].clamp(max=self.match_init)

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -self.INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        result = AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        if self.debug:
            self._debug(result, priors, decoded_bboxes, gt_bboxes)
        return result

    def _debug(self, assign_result: AssignResult, priors, decoded_bboxes,
               gt_bboxes):
        import matplotlib.pyplot as plt
        plt.figure(dpi=300)
        pos = assign_result.gt_inds.gt(0)
        pos_gt_bboxes = gt_bboxes[assign_result.gt_inds[pos] - 1]
        pos_decoded_bboxes = decoded_bboxes[pos]
        pos_decoded_bboxes = type(pos_gt_bboxes)(pos_decoded_bboxes)

        pos_gt_bboxes_bevs = pos_gt_bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        pos_gt_bboxes_x = pos_gt_bboxes_bevs[..., 0].cpu().numpy().T
        pos_gt_bboxes_cx = pos_gt_bboxes_x[:4, ...].mean(axis=0)
        pos_gt_bboxes_y = pos_gt_bboxes_bevs[..., 1].cpu().numpy().T
        pos_gt_bboxes_cy = pos_gt_bboxes_y[:4, ...].mean(axis=0)
        pos_decoded_bboxes_bev = pos_decoded_bboxes.corners[
                                 :, [0, 3, 7, 4, 0], :2]
        pos_decoded_bboxes_x = pos_decoded_bboxes_bev[..., 0].cpu().numpy().T
        pos_decoded_bboxes_y = pos_decoded_bboxes_bev[..., 1].cpu().numpy().T

        pos_priors = priors[pos]

        pos_priors_x = pos_priors[..., 0].cpu().numpy()
        pos_priors_y = pos_priors[..., 1].cpu().numpy()
        match_x = np.stack([pos_gt_bboxes_cx, pos_priors_x])
        match_y = np.stack([pos_gt_bboxes_cy, pos_priors_y])

        plt.plot(pos_gt_bboxes_x, pos_gt_bboxes_y, 'r', lw=1)
        plt.plot(pos_decoded_bboxes_x, pos_decoded_bboxes_y, 'g', lw=1)
        plt.plot(pos_priors_x, pos_priors_y, 'b.', ms=2)
        plt.plot(match_x, match_y, lw=1)
        plt.axis('equal')
        plt.show()

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        xyz = torch.cat([priors[:, :2], priors.new_zeros(priors.size(0), 1)],
                        dim=-1)
        gt_bev = gt_bboxes.clone()
        gt_bev[..., 2] = -self.INF
        gt_bev[..., 5] = 2 * self.INF
        is_in_gts = points_in_boxes_all(
            xyz.unsqueeze(0), gt_bev.unsqueeze(0))[0] > 0
        is_in_gts_any = is_in_gts.any(dim=1)

        xy = priors[:, :2].unsqueeze(1)

        gt_xy = gt_bboxes[..., :2].unsqueeze(0)

        is_in_cts = (xy - gt_xy).abs().max(-1).values < self.center_radius
        is_in_cts_any = is_in_cts.any(dim=1)

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_any | is_in_cts_any

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
                is_in_gts[is_in_gts_or_centers, :]
                & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        topk = min(self.candidate_topk, pairwise_ious.shape[0])
        topk_ious, _ = torch.topk(pairwise_ious, topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
