voxel_size_cart = [0.16, 0.16, 4]
point_cloud_range_cart = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size_cyn = [0.0025, 0.04, 100]
point_cloud_range_cyn = [-1.58, -3, 0, 1.58, 1, 100]

model = dict(
    type='PillarODCenterPoint',
    pts_voxel_layer=dict(
        view=['cartesian', 'cylindrical'],
        voxel_size=[voxel_size_cart, voxel_size_cyn],
        point_cloud_range=[point_cloud_range_cart, point_cloud_range_cyn],
        max_num_points=[-1, -1]),
    pts_voxel_encoder=dict(
        type='PillarMVFFeatureNet',
        in_channels=4,
        feat_channels=64,
        voxel_size=[voxel_size_cart, voxel_size_cyn],
        point_cloud_range=[point_cloud_range_cart, point_cloud_range_cyn],
        norm1d_cfg=dict(type='BN1d', eps=1e-3, momentum=0.03),
        norm2d_cfg=dict(type='BN2d', eps=1e-3, momentum=0.03)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.03)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.03)),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=2, class_names=['Car'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoderRev',
            out_size_factor=2,
            voxel_size=voxel_size_cart[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3,
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.03)),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.03)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[496, 432, 1],
            voxel_size=voxel_size_cart,
            out_size_factor=2,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-10, -49.68, -10, 79.12, 49.68, 10],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[0.85, 0.175, 4],
            score_threshold=0.1,
            pc_range=[0, -39.68],
            out_size_factor=2,
            voxel_size=voxel_size_cart[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))
