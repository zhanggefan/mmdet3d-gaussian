_base_ = './hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'

data = dict(samples_per_gpu=12)

optimizer = dict(lr=0.001)
