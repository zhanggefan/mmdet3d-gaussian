from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.apis import init_model
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel
import mmdet3d_gwd
import open3d as o3d
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import argparse
import os.path as osp
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--pth', help='train checkpoint path', default=None)
    parser.add_argument('--out', help='inference result', default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.test)

    model = None
    loader = None
    if args.pth:
        model = init_model(cfg, args.pth)
        model = revert_sync_batchnorm(model)
        model = MMDataParallel(model, device_ids=[0])
        loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    out = None
    if args.out:
        out = mmcv.load(args.out)

    def renderbox(box3d, labels, color=None):
        clr_map = plt.get_cmap('tab10').colors
        corners = box3d.corners
        cores = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (8, 4), (8, 5), (8, 6), (8, 7)
        ]
        ret = None
        for corners_i, label_i in zip(corners, labels):
            corners_i = corners_i.numpy().astype(np.float64)
            frontcenter = corners_i[[4, 5, 6, 7]].mean(axis=0, keepdims=True)
            heading = corners_i[4] - corners_i[0]
            frontcenter += 0.3 * heading / np.linalg.norm(heading)
            corners_i = np.concatenate((corners_i, frontcenter), axis=0)
            corners_i = o3d.utility.Vector3dVector(corners_i)
            corners_i = o3d.geometry.PointCloud(points=corners_i)

            box = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                corners_i,
                corners_i,
                cores)
            if color is None:
                box.paint_uniform_color(clr_map[label_i % len(clr_map)])
            else:
                box.paint_uniform_color(color)
            if ret is None:
                ret = box
            else:
                ret += box

        return ret

    def rendergroundplane(plane, griddim=5, gridpts=21):
        a = np.linspace(-gridpts // 2, gridpts // 2, gridpts) * griddim
        b = np.linspace(0, gridpts - 1, gridpts) * griddim
        aa, bb = np.meshgrid(a, b)
        plane_x, plane_y, plane_z, plane_off = plane
        dir1 = np.array([0, plane_z, -plane_y])
        dir2 = np.array(
            [plane_y * plane_y + plane_z * plane_z, -plane_x * plane_y,
             -plane_x * plane_z])
        off_dir = -plane_off * np.array([plane_x, plane_y, plane_z])
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)
        dirmat = np.stack((dir1, dir2), axis=0)
        pts = np.stack((aa, bb), axis=-1).reshape(-1, 2)
        pts = pts @ dirmat + off_dir
        pts = o3d.utility.Vector3dVector(pts)
        pts = o3d.geometry.PointCloud(points=pts)
        cores = [(p * gridpts + i, p * gridpts + j) for i, j in
                 zip(range(gridpts - 1), range(1, gridpts)) for p in
                 range(gridpts)]
        cores += [(p + i * gridpts, p + j * gridpts) for i, j in
                  zip(range(gridpts - 1), range(1, gridpts)) for p in
                  range(gridpts)]
        grid = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pts,
            pts,
            cores)
        grid.paint_uniform_color(((0.5), (0.5), (0.5)))
        return grid

    progress_bar = mmcv.ProgressBar(len(dataset))

    def vis_iter_pth(loader, model):
        for i, data in enumerate(loader):
            with torch.no_grad():
                det = model(return_loss=False, rescale=True, **data)
            yield i, loader.dataset[i], det[0]

    def vis_iter_out(dataset, out):
        for i, data in enumerate(dataset):
            det = out[i]
            yield i, data, det

    if out:
        vis_iter = vis_iter_out(dataset, out)
    elif model:
        vis_iter = vis_iter_pth(loader, model)

    def key_cbk(vis: o3d.visualization.Visualizer):
        try:
            idx, data, det = next(vis_iter)
        except StopIteration:
            return True
        box3d = data['gt_bboxes_3d'][0].data
        names = data['gt_labels_3d'][0].data
        points = data['points'][0].data

        det_box3d = det['boxes_3d']
        det_names = det['labels_3d']
        det_scores = det['scores_3d']

        # v = names.le(2) & names.ge(0)
        # box3d = box3d[v]
        # names = names[v]

        v = det_scores.ge(0.3)
        det_box3d = det_box3d[v]
        det_names = det_names[v]

        clr_map = plt.get_cmap('gist_rainbow')
        xyz = points[:, :3].numpy().astype(np.float64)
        clr = clr_map(points[:, 3])[:, :3]

        points = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(xyz))
        points.colors = o3d.utility.Vector3dVector(clr)
        gt_box = renderbox(box3d, names, [1, 0, 0])
        if len(det_box3d):
            det_box = renderbox(det_box3d, det_names, [0, 1, 0])
        vis.clear_geometries()
        vis.add_geometry(gt_box, idx == 0)
        if len(det_box3d):
            vis.add_geometry(det_box, idx == 0)
        vis.add_geometry(points, idx == 0)
        # if 'plane' in data:
        #     plane = data['plane'].data
        #     plane = rendergroundplane(plane)
        #     vis.add_geometry(plane, idx == 0)

        # ctl = vis.get_view_control()
        # ctl.set_front([[0], [-1], [1]])
        # ctl.set_lookat([[30], [0], [0]])
        # ctl.set_zoom(0.3)

        progress_bar.update()
        # vis.capture_screen_image(
        #     f'out/{osp.splitext(osp.split(args.out)[-1])[0]}/{idx}.png', True)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(" "), key_cbk)
    vis.create_window(width=1080, height=720)
    op = vis.get_render_option()
    # op.background_color = np.array([1., 1., 1.])
    op.background_color = np.array([0., 0., 0.])
    op.point_size = 2.0
    if key_cbk(vis):
        return
    else:
        vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
