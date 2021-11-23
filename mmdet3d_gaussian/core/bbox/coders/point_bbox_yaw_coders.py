import torch
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox import BaseBBoxCoder
import numpy as np


@BBOX_CODERS.register_module()
class PointBBoxYawCoder(BaseBBoxCoder):
    @property
    def code_size(self):
        return 9

    def encode(self, target_boxes):
        yaw = target_boxes[..., 6]
        dir = torch.stack((yaw.sin(), yaw.cos()), dim=-1)
        others = target_boxes[..., 7:]
        return torch.cat((target_boxes[..., :7], dir, others), dim=-1)

    def decode(self, priors, preds, correct_yaw=True):
        """Decode bboxes.
        Args:
            priors (torch.Tensor): center locations of shape [B, K, 3],
                where K is the number of predictions to decode per sample,
                typically: {dx, dy, z, w, h, l, sin(dir), cos(dir), (...)}
            preds (torch.Tensor): box predictions of shape [B, K, N],
                where N is the number of features per box.
            correct_yaw: bool
                does the yaw need to be corrected by dir
        Returns:
            list[dict]: Decoded boxes.
        """
        xy = preds[..., :2] * priors[..., 2:] + priors[..., :2]
        z = preds[..., 2]
        dim = preds[..., 3:6].exp()
        dim = torch.cat((dim[..., :2] * priors[..., 2:], dim[..., 2:]), -1)
        yaw = preds[..., 6]

        if correct_yaw:
            with torch.no_grad():
                dir_sine = preds[..., 7]
                dir_cosine = preds[..., 8]
                dir = torch.atan2(dir_sine, dir_cosine)
                num_rot90 = torch.floor((dir - yaw) / (np.pi / 2) + 0.5)
                no_swap_wh = (num_rot90.long() % 2 == 0)
            yaw = yaw + num_rot90 * (np.pi / 2)

            dim = dim.where(no_swap_wh.unsqueeze(-1), dim[..., [1, 0, 2]])

        others = preds[..., 9:]

        return torch.cat((xy, z.unsqueeze(-1), dim, yaw.unsqueeze(-1), others),
                         dim=-1)
