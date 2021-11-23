import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class CenterPointBBoxCoderRev(BaseBBoxCoder):
    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 code_size=9,
                 norm_bbox=True):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.code_size = code_size
        self.norm_bbox = norm_bbox

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].
            K (int): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()
        # top k per class first
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.float() // width).long()
        topk_xs = (topk_inds % width).long()
        # top k for all class
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind // K).long()
        topk_ys = topk_ys.view(batch, -1).gather(dim=1, index=topk_ind)
        topk_xs = topk_xs.view(batch, -1).gather(dim=1, index=topk_ind)

        return topk_score, topk_clses, topk_ys, topk_xs

    def select_best(self, scores, preds, topk):
        batch, cat, _, _ = scores.size()
        scores, clses, ys, xs = self._topk(scores, K=topk)
        locs = torch.stack((xs, ys), dim=-1)
        preds = preds.permute(0, 2, 3, 1)
        preds = torch.stack([preds[b, ys[b], xs[b]] for b in range(batch)],
                            dim=0)
        return scores, clses, locs, preds

    def encode(self, target_boxes):
        pc_range = self.pc_range
        voxel_size = self.voxel_size
        out_size_factor = self.out_size_factor

        x_unscale = (target_boxes[..., 0] - pc_range[0]) / voxel_size[
            0] / out_size_factor
        y_unscale = (target_boxes[..., 1] - pc_range[1]) / voxel_size[
            1] / out_size_factor
        x_res = x_unscale - x_unscale.floor()
        y_res = y_unscale - y_unscale.floor()
        z = target_boxes[..., 2]

        xyz = torch.stack((x_res, y_res, z), dim=-1)

        dim = target_boxes[..., 3:6]

        if self.norm_bbox:
            dim = dim.log()
        rot = target_boxes[..., 6]
        rot = torch.stack((rot.sin(), rot.cos()), dim=-1)
        others = target_boxes[..., 7:]

        return torch.cat((xyz, dim, rot, others), dim=-1)

    def decode(self, locs, preds):
        """Decode bboxes.

        Args:
            locs (torch.Tensor): center locations of shape [B, K, 2],
                where K is the number of predictions to decode per sample,
                typically: {dx, dy, z, w, h, l, sin(rot), cos(rot), (...)}
            preds (torch.Tensor): box predictions of shape [B, K, N],
                where N is the number of features per box.
        Returns:
            list[dict]: Decoded boxes.
        """
        x = (preds[..., 0] + locs[..., 0]) * self.out_size_factor * \
            self.voxel_size[0] + self.pc_range[0]
        y = (preds[..., 1] + locs[..., 1]) * self.out_size_factor * \
            self.voxel_size[1] + self.pc_range[1]
        z = preds[..., 2]
        dim = preds[..., 3:6]
        if self.norm_bbox:
            dim = dim.exp()
        rot_sine = preds[..., 6]
        rot_cosine = preds[..., 7]
        rot = torch.atan2(rot_sine, rot_cosine)
        others = preds[..., 8:]

        return torch.cat(
            (x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1), dim,
             rot.unsqueeze(-1), others), dim=-1)
