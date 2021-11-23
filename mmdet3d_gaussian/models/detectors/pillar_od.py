from mmdet.models import DETECTORS
from mmdet3d.ops.voxel.voxelize import Voxelization
from .detectors_rev import CenterPointRev, VoxelNetRev
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
import torch
import torch.nn.functional as F


@DETECTORS.register_module()
class PillarODCenterPoint(CenterPointRev):
    def __init__(self, **kwargs):
        voxelizers_cfg = kwargs.pop('pts_voxel_layer')
        super(PillarODCenterPoint, self).__init__(**kwargs)
        self.pts_voxel_views = []
        self.pts_voxel_layers = []
        views = voxelizers_cfg.pop('view')
        for i, v in enumerate(views):
            self.pts_voxel_views.append(v)
            cfg = {k: v[i] for k, v in voxelizers_cfg.items()}
            self.pts_voxel_layers.append(Voxelization(**cfg))
        self.pts_voxel_layers = torch.nn.ModuleList(self.pts_voxel_layers)

    @torch.no_grad()
    @force_fp32()
    def to_cartesian(self, points):
        return points

    @torch.no_grad()
    @force_fp32()
    def to_cylindrical(self, points):
        z = points[:, 2]
        phi = torch.atan2(points[:, 1], points[:, 0])
        rho = points[:, :2].norm(p=2, dim=-1)
        cylinder = torch.stack((phi, z, rho), dim=-1)
        return torch.cat((cylinder, points[:, 3:]), dim=-1)

    @torch.no_grad()
    @force_fp32()
    def to_spherical(self, points):
        yaw = torch.atan2(points[:, 1], points[:, 0])
        rho = points[:, :3].norm(p=2, dim=-1)
        pitch = torch.asin(points[:, 2] / rho)
        sphere = torch.stack((yaw, pitch, rho), dim=-1)
        return torch.cat((sphere, points[:, 3:]), dim=-1)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        return multi_apply(self.voxelize_view,
                           self.pts_voxel_views,
                           self.pts_voxel_layers,
                           points=points)

    @torch.no_grad()
    @force_fp32()
    def voxelize_view(self, view, voxelizer, points):
        points = [getattr(self, f'to_{view}')(res) for res in points]
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = voxelizer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_metas)
        batch_size = coors[0][-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x


@DETECTORS.register_module()
class PillarODVoxelNet(VoxelNetRev):
    def __init__(self, **kwargs):
        voxelizers_cfg = kwargs.pop('voxel_layer')
        dummy_voxelizer_cfg = {k: v[0] for k, v in voxelizers_cfg.items() if
                               k != 'view'}
        super(PillarODVoxelNet, self).__init__(**kwargs,
                                               voxel_layer=dummy_voxelizer_cfg)
        self.voxel_views = []
        self.voxel_layers = []
        views = voxelizers_cfg.pop('view')
        for i, v in enumerate(views):
            self.voxel_views.append(v)
            cfg = {k: v[i] for k, v in voxelizers_cfg.items()}
            self.voxel_layers.append(Voxelization(**cfg))
        self.voxel_layers = torch.nn.ModuleList(self.voxel_layers)

    @torch.no_grad()
    @force_fp32()
    def to_cartesian(self, points):
        return points

    @torch.no_grad()
    @force_fp32()
    def to_cylindrical(self, points):
        z = points[:, 2]
        phi = torch.atan2(points[:, 1], points[:, 0])
        rho = points[:, :2].norm(p=2, dim=-1)
        cylinder = torch.stack((phi, z, rho), dim=-1)
        return torch.cat((cylinder, points[:, 3:]), dim=-1)

    @torch.no_grad()
    @force_fp32()
    def to_spherical(self, points):
        yaw = torch.atan2(points[:, 1], points[:, 0])
        rho = points[:, :3].norm(p=2, dim=-1)
        pitch = torch.asin(points[:, 2] / rho)
        sphere = torch.stack((yaw, pitch, rho), dim=-1)
        return torch.cat((sphere, points[:, 3:]), dim=-1)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        return multi_apply(self.voxelize_view,
                           self.voxel_views,
                           self.voxel_layers,
                           points=points)

    @torch.no_grad()
    @force_fp32()
    def voxelize_view(self, view, voxelizer, points):
        points = [getattr(self, f'to_{view}')(res) for res in points]
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = voxelizer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_feat(self, points, img_metas=None):
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.voxel_encoder(
            voxels, coors)
        batch_size = coors[0][-1, 0] + 1
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x
