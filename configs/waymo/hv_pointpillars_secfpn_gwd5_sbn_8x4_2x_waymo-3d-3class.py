_base_ = './hv_pointpillars_secfpn_sbn_8x4_2x_waymo-3d-3class.py'

model = dict(
    pts_bbox_head=dict(
        type='GDAnchor3DHead',
        loss_bbox=dict(
            type='GDLoss', loss_type='gwd3d',
            center_offset=(0, 0, 0.5), fun='log1p', tau=0.0, alpha=1.0,
            loss_weight=5.0)),
    train_cfg=dict(
        code_weight=[0., 0., 0., 0., 0., 0., 0.],
        decode_weight=1,
    )
)
