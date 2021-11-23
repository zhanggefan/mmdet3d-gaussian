_base_ = './centerpoint_02pillar_second_secfpn_8x4_cyclic_20e_nus.py'

model = dict(
    pts_bbox_head=dict(
        type='CenterGDHead',
        common_heads=dict(_delete_=True, reg=(2, 2), height=(1, 2), dim=(3, 2),
                          yaw=(1, 2), dir=(2, 2), vel=(2, 2)),
        bbox_coder=dict(type='CenterPointBBoxYawCoder'),
        loss_gd=dict(type='GDLoss', loss_type='gwd3d', reduction='mean',
                     fun='log1p', tau=0.0, alpha=1.0, loss_weight=5.0)),
    train_cfg=dict(
        pts=dict(
            code_weights=[1.0, 1.0, 0.2, 0.2])))  # for dir and vel only
