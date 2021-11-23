_base_ = './hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'

model = dict(
    type='DynamicVoxelNet',
    voxel_layer=dict(
        max_num_points=-1,
        max_voxels=-1
    ),
    voxel_encoder=dict(
        type='DynamicPillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_cluster_center=False,
        with_cluster_center_offset=True,
        with_covariance=False,
        with_voxel_center=False,
        with_voxel_point_count=False,
        with_voxel_center_offset=True))

data = dict(samples_per_gpu=12)

optimizer = dict(lr=0.001)
