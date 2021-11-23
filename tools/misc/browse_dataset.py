from mmcv import Config
from mmdet3d.datasets import build_dataset
import mmdet3d_gaussian
import open3d as o3d
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.train)

    def renderbox(box3d, labels):
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
            box.paint_uniform_color(clr_map[label_i % len(clr_map)])
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
    dataset = enumerate(dataset)

    def key_cbk(vis: o3d.visualization.Visualizer):
        try:
            idx, data = next(dataset)
        except StopIteration:
            return True
        box3d = data['gt_bboxes_3d'].data
        names = data['gt_labels_3d'].data
        points = data['points'].data
        clr_map = plt.get_cmap('gist_rainbow')
        xyz = points[:, :3].numpy().astype(np.float64)
        clr = clr_map(points[:, 3])[:, :3]

        points = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(xyz))
        points.colors = o3d.utility.Vector3dVector(clr)
        rdbox = renderbox(box3d, names)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.clear_geometries()
        vis.add_geometry(rdbox, idx == 0)
        vis.add_geometry(points, idx == 0)
        vis.add_geometry(axis, idx == 0)
        if 'plane' in data:
            plane = data['plane'].data
            plane = rendergroundplane(plane)
            vis.add_geometry(plane, idx == 0)
        progress_bar.update()
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(" "), key_cbk)
    vis.create_window()
    op = vis.get_render_option()
    op.background_color = np.array([0., 0., 0.])

    if key_cbk(vis):
        return
    else:
        vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
