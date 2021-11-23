from mmdet.models import DETECTORS
from mmdet3d.models.detectors.two_stage import TwoStage3DDetector
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet3d.models import builder
from .utils import ShowResultMixin
from mmcv.runner import force_fp32
import torch
import torch.nn.functional as F


@DETECTORS.register_module()
class PVRCNN(ShowResultMixin, TwoStage3DDetector):
    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 voxel_middle_encoder,
                 keypoints_encoder,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStage3DDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.voxel_middle_encoder = builder.build_middle_encoder(
            voxel_middle_encoder)
        self.keypoints_encoder = builder.build_middle_encoder(
            keypoints_encoder)

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        voxel_dict = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        feats_dict = self.voxel_middle_encoder(voxel_features,
                                               voxel_dict['coors'],
                                               batch_size)
        x = self.backbone(feats_dict['spatial_features'])
        if self.with_neck:
            neck_feats = self.neck(x)
            feats_dict.update({'neck_features': neck_feats})

        keypoints_feats_dict = self.keypoints_encoder(
            points=points, coors=voxel_dict['coors'],
            voxel_encode_features=feats_dict['encode_features'],
            bev_encode_features=feats_dict['spatial_features'])
        feats_dict.update(keypoints_feats_dict)

        return feats_dict

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        voxel_dict = dict(
            voxels=voxels,
            num_points=num_points,
            coors=coors_batch)
        return voxel_dict

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None,
                      proposals=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        feats_dict = self.extract_feat(points, img_metas)

        losses = dict()

        if self.with_rpn:
            rpn_outs = self.rpn_head(feats_dict['neck_features'])
            rpn_loss_inputs = rpn_outs + (gt_bboxes_3d, gt_labels_3d,
                                          img_metas)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(feats_dict, img_metas,
                                                 proposal_list, gt_bboxes_3d,
                                                 gt_labels_3d)

        losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, proposals=None, rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        feats_dict = self.extract_feat(points, img_metas)

        if self.with_rpn:
            rpn_outs = self.rpn_head(feats_dict['neck_features'])
            proposal_cfg = self.test_cfg.rpn
            bbox_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*bbox_inputs)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(feats_dict, img_metas, proposal_list)
