_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# data settings
data = dict(train=dict(times=1, dataset=dict(load_interval=1)))

fp16 = dict(loss_scale='dynamic')
