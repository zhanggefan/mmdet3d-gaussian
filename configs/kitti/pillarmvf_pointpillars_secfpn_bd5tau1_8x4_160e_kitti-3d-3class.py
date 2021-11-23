_base_ = './pillarmvf_pointpillars_secfpn_8x4_160e_kitti-3d-3class.py'

model = dict(
    bbox_head=dict(
        type='GDAnchor3DHead',
        loss_decoded_bbox=dict(
            type='GDLoss', loss_type='bd3d', center_offset=(0, 0, 0.5),
            fun='log1p', tau=1.0, alpha=1.0, loss_weight=5.0)),
    train_cfg=dict(
        code_weight=[0., 0., 0., 0., 0., 0., 0.],
        decode_weight=1,
    )
)

data = dict(samples_per_gpu=8)

optimizer = dict(lr=0.00075)
