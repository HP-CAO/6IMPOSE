import os
from lib.data.unity.unity import Unity

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import cv2
import numpy as np
import tensorflow as tf
from lib.data.utils import compute_normal_map, get_bbox_from_mask, load_mesh, rescale_image_bbox
from lib.data.dataset import NoDepthError, NoMaskError, NoRtError
from lib.data.augmenter import add_background_depth, augment_depth, augment_rgb, rotate_datapoint
from lib.monitor.visualizer import draw_single_bbox, project2img, visualize_normal_map
import json
import math
from tqdm import tqdm
import re
import scipy.stats as stats
import matplotlib.pyplot as plt


def get_psd_of_depth_list(depth_list):
    avg_psd = []
    for depth in depth_list:
        psd = np.fft.fft2((depth - np.mean(depth)) / np.std(depth))  # [480, 640]
        psd = np.abs(psd) ** 2
        psd = psd.flatten()

        avg_psd.append(psd)

    psd = np.array(avg_psd)
    psd = np.mean(psd, 0)

    kfreq_x = np.fft.fftfreq(depth.shape[0]) * depth.shape[0]
    kfreq_y = np.fft.fftfreq(depth.shape[1]) * depth.shape[1]

    kfreq2D = np.meshgrid(kfreq_y, kfreq_x)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()

    kbins = np.arange(0.5, depth.shape[0] // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(knrm, psd, statistic="mean", bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return kvals, Abins


def sobel(img):
    scale = 1
    delta = 0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


class Mode:
    bbox = False
    mask = False
    depth = False
    statistics = False
    normals = False
    pose = False
    psd = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', default='', nargs='+',
                        help='format as <data_set>/<data_name>/<cls_type>. Add /aug to perform augmentation on that dataset')
    parser.add_argument('--mode', default='bbox|mask|depth|normals|statistics|pose',
                        help='[bbox|mask|depth|normals|statistics|pose]: define what to inspect. join multiple ops with /')
    parser.add_argument('--num_imgs', default=50, type=int, help='Number of images to inspect')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--no-vis', dest='vis', action='store_false')
    parser.set_defaults(vis=True)
    args = parser.parse_args()

    mode = Mode()
    for key in Mode.__dict__.keys():
        if not key.startswith('__'):
            setattr(mode, key, key in args.mode)

    from lib.factory import DatasetFactory
    factory = DatasetFactory()

    preview_size = (480, 640)

    datasets = {}

    inds = None

    for ds in args.datasets:
        aug = 1 if '/aug' in ds else 0
        datasets[ds] = factory.build_dataset(*ds.split('/')[:3], use_preprocessed=False, size_all=1000, train_size=500,
                                             augment_per_image=aug)

        if inds is None:
            inds = [int(re.search('[0-9]+', x).group(0)) for x in
                    os.listdir(os.path.join(datasets[ds].cls_root, 'rgb'))]
        else:
            ds_inds = [int(re.search('[0-9]+', x).group(0)) for x in
                       os.listdir(os.path.join(datasets[ds].cls_root, 'rgb'))]
            inds = [x for x in inds if x in ds_inds]  # filter to only show indices that are available everywhere

    try:
        inds = np.random.choice(inds, args.num_imgs, replace=False)
    except ValueError:
        print(f"There are less images available than requested. ({len(inds)} < {args.num_imgs})")
        exit(-1)

    all_data = {ds_name: [] for ds_name in args.datasets}
    meshes = {ds_name: {} for ds_name in args.datasets}

    print("Gathering data...")
    for idx in tqdm(inds):
        for ds_name, ds in datasets.items():
            data = {}

            # --- DATA ACQUISITION ---

            rgb = ds.get_rgb(idx)

            try:
                depth = ds.get_depth(idx)
            except NoDepthError:
                depth = None
            try:
                mask = ds.get_mask(idx)
            except NoMaskError:
                mask = None
            try:
                Rt = ds.get_RT_list(idx)
            except NoRtError:
                Rt = None
            bboxes = ds.get_gt_bbox(idx)

            # --- AUGMENTATION ---
            if ds.if_augment:
                if depth is not None and Rt is not None:
                    depth_tensor = tf.expand_dims(depth, 0)  # add batch
                    depth_tensor = tf.cast(tf.expand_dims(depth_tensor, -1), tf.float32)  # channel

                    obj_pos = Rt[0][0][:3, 3] if ds.cls_type != 'all' else None

                    depth_tensor = add_background_depth(depth_tensor, obj_pos=obj_pos,
                                                        camera_matrix=ds.data_config.intrinsic_matrix)

                    depth_tensor = augment_depth(depth_tensor)
                    depth = tf.squeeze(depth_tensor).numpy()

                if mask is not None:
                    # only allow rotation when mask is available -> then bboxes can be derived from mask
                    rgb, mask, depth, Rt = rotate_datapoint(img_likes=[rgb, mask, depth], Rt=Rt)
                    bboxes = []
                    if ds.cls_type == 'all':
                        for cls, gt_mask_value in ds.data_config.mask_ids.items():
                            bbox = get_bbox_from_mask(mask, gt_mask_value)
                            if bbox is None:
                                continue
                            bbox = list(bbox)
                            bbox.append(ds.data_config.obj_dict[cls])
                            bboxes.append(bbox)
                    else:
                        bbox = get_bbox_from_mask(mask, gt_mask_value=255)
                        if bbox is not None:
                            bbox = list(bbox)
                            bbox.append(ds.cls_id)
                            bboxes.append(bbox)
                    bboxes = np.array(bboxes)

                rgb = (augment_rgb(rgb.astype(np.float32) / 255.) * 255).astype(np.uint8)

            intrinsic_matrix = ds.data_config.intrinsic_matrix.copy()

            # --- RGB ---
            rgb = rgb.astype(np.uint8)
            data['rgb'] = rgb

            # --- MASK ---
            if mode.mask:
                if mask is not None:
                    # max_mask_id = np.max(list(ds.data_config.mask_ids.values()))
                    # vis_mask = (mask/max_mask_id*255).astype(np.uint8)
                    vis_mask = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
                else:
                    vis_mask = np.zeros_like(rgb)
                data['mask'] = cv2.cvtColor(vis_mask, cv2.COLOR_BGR2RGB)

            # --- BBOX ---
            if mode.bbox:
                max_cls_id = np.max(list(ds.data_config.obj_dict.values()))
                data['bbox'] = []

                bgr_bbox = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                for box in bboxes:
                    x1, y1, x2, y2, cls_id = box
                    data['bbox'].append({'cls': cls_id, 'box': (x1, y1, x2, y2)})
                    color = cv2.applyColorMap(np.array(cls_id / max_cls_id * 255).astype(np.uint8),
                                              cv2.COLORMAP_TURBO).flatten()
                    color = tuple((int(c) for c in color))
                    label = f"{ds.data_config.id2obj_dict[cls_id]}"
                    draw_single_bbox(bgr_bbox, (x1, y1, x2, y2), label, color)

                data['bboxes'] = cv2.cvtColor(bgr_bbox, cv2.COLOR_BGR2RGB)

            # --- POSE ---
            if mode.pose and Rt is not None:

                cls_types = [ds.data_config.cls_type]
                if cls_types[0] == 'all':
                    cls_types = list(ds.data_config.obj_dict.keys())

                for cls_type in cls_types:
                    if cls_type in meshes[ds_name].keys():
                        continue

                    if cls_type == 'all':
                        continue
                    try:
                        mesh_path = ds.data_config.mesh_paths[cls_type]
                    except KeyError:
                        continue

                    mesh_points = load_mesh(mesh_path, scale=ds.data_config.mesh_scale, n_points=500)

                    if isinstance(ds, Unity):
                        mesh_points[:, 0] *= 1

                    kpts_path = os.path.join(ds.data_config.kps_dir, "{}/farthest.txt".format(cls_type))
                    corner_path = os.path.join(ds.data_config.kps_dir, "{}/corners.txt".format(cls_type))
                    key_points = np.loadtxt(kpts_path)
                    center = [np.loadtxt(corner_path).mean(0)]
                    mesh_kpts = np.concatenate([key_points, center], axis=0)

                    meshes[ds_name][cls_type] = (mesh_points, mesh_kpts)

                bgr_pose = cv2.cvtColor((rgb.copy()).astype(np.float32), cv2.COLOR_RGB2BGR)
                max_cls_id = np.max(list(ds.data_config.obj_dict.values()))

                for Rt, cls_id in Rt:

                    # img_gt_pts_seg = vis_pts_semantics(rgb.copy(), label_segs, sampled_index)
                    # img_gt_kpts, pts_2d_gt = vis_gt_kpts(rgb.copy(), mesh_kpts, RT_gt, xy_offset, cam_intrinsic)
                    # img_gt_offset = vis_offset_value(sampled_index, label_segs, [kpts_targ_offst],
                    #                                [ctr_targ_offst], pts_2d_gt, img_shape=rgb.shape)

                    # log prediction
                    try:
                        mesh_points, mesh_kpts = meshes[ds_name][ds.data_config.id2obj_dict[cls_id]]
                    except KeyError:
                        continue

                    color = cv2.applyColorMap(np.array(cls_id / max_cls_id * 255).astype(np.uint8),
                                              cv2.COLORMAP_TURBO).flatten()
                    color = tuple((int(c) for c in color))

                    bgr_pose = project2img(mesh_points, Rt, bgr_pose, intrinsic_matrix,
                                           ds.data_config.camera_scale, color, (0, 0)) * 255

                data['pose'] = cv2.cvtColor(bgr_pose.astype(np.uint8), cv2.COLOR_BGR2RGB)

                # img_pre_proj = cv2.putText(img_pre_proj, "ADD: {:.5f} ADDS: {:.5f}".format(add_score, adds_score),
                #                        (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 255), 1)

                # pre_segs = np.argmax(seg_pre.numpy(), axis=-1).squeeze()

                # img_pre_pts_seg = vis_pts_semantics(rgb.copy(), pre_segs, sampled_index)
                # img_pre_ktps, pts_2d_pre = vis_pre_kpts(rgb.copy(), kpts_voted, xy_offset, cam_intrinsic)
                # img_pre_offset = vis_offset_value(sampled_index, pre_segs, kp_pre_ofst.numpy(),
                #                                cp_pre_ofst.numpy(), pts_2d_gt, pts_2d_pre, rgb.shape)

            # --- DEPTH ---
            if mode.depth:
                if depth is not None:
                    vis_depth = cv2.convertScaleAbs(depth, alpha=255 / np.max(depth))
                    vis_depth = cv2.applyColorMap(vis_depth, cv2.COLORMAP_JET)
                else:
                    vis_depth = np.zeros_like(rgb)
                data['depth'] = cv2.cvtColor(vis_depth, cv2.COLOR_BGR2RGB)

            # --- NORMALS ---
            if mode.normals:
                if depth is not None:
                    __depth = tf.expand_dims(tf.expand_dims(depth.astype(np.float32), axis=-1), axis=0)
                    camera_matrix = ds.data_config.intrinsic_matrix
                    normal_map = compute_normal_map(__depth, camera_matrix.astype(np.float32))
                    normals = visualize_normal_map(tf.squeeze(normal_map))
                else:
                    normals = np.zeros_like(rgb)
                data['normals'] = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)

            # ---- PSD ------------
            if mode.psd:
                if depth is not None:
                    data['depth_data'] = depth

            all_data[ds_name].append(data)

    # global statistics here
    output = {}
    if mode.bbox:
        box_data = {ds_name: {'counts': {}} for ds_name in datasets.keys()}

        for ds_name in datasets.keys():
            widths = {}
            heights = {}
            sizes = {}

            img_w, img_h = all_data[ds_name][0]['rgb'].shape[:2]

            all_boxes = [x['bbox'] for x in all_data[ds_name]]  # [{'cls':cls, 'box':box}, ...]
            all_boxes = [x for img_boxes in all_boxes for x in img_boxes]
            for single_box in all_boxes:
                w = int(single_box['box'][2] - single_box['box'][0]) / img_w
                h = int(single_box['box'][3] - single_box['box'][1]) / img_h
                size = math.sqrt(w * h)  # according to https://arxiv.org/pdf/2107.04259.pdf

                widths.setdefault(single_box['cls'], [])
                widths[single_box['cls']].append(w)

                heights.setdefault(single_box['cls'], [])
                heights[single_box['cls']].append(h)

                sizes.setdefault(single_box['cls'], [])
                sizes[single_box['cls']].append(size)

                box_data[ds_name]['counts'].setdefault(single_box['cls'], 0)
                box_data[ds_name]['counts'][single_box['cls']] += 1

            box_data[ds_name]['mean_width'] = {cls: float(np.mean(cls_widths)) for cls, cls_widths in widths.items()}
            box_data[ds_name]['mean_height'] = {cls: float(np.mean(cls_heights)) for cls, cls_heights in
                                                heights.items()}
            box_data[ds_name]['mean_size'] = {cls: float(np.mean(cls_sizes)) for cls, cls_sizes in sizes.items()}

            box_data[ds_name]['std_width'] = {cls: float(np.std(cls_widths)) for cls, cls_widths in widths.items()}
            box_data[ds_name]['std_height'] = {cls: float(np.std(cls_heights)) for cls, cls_heights in heights.items()}
            box_data[ds_name]['std_size'] = {cls: float(np.std(cls_sizes)) for cls, cls_sizes in sizes.items()}

        output['bbox_statistics'] = box_data

    if mode.statistics:
        stats = {ds_name: {} for ds_name in datasets.keys()}

        for ds_name in datasets.keys():
            all_imgs = np.array([x['rgb'] for x in all_data[ds_name]])
            all_grad_imgs = np.array([sobel(img) for img in all_imgs])

            all_rgbs = np.reshape(all_imgs, (-1, 3))
            all_hsvs = cv2.cvtColor(np.expand_dims(all_rgbs, 0), cv2.COLOR_RGB2HSV)[0, :]
            all_grads = all_grad_imgs.flatten()

            np2tup = lambda nump_arr: tuple((float(x) for x in nump_arr))

            stats[ds_name]['mean_rgb'] = np2tup(np.mean(all_rgbs, 0))
            stats[ds_name]['mean_hsv'] = np2tup((np.mean(all_hsvs, 0)))
            stats[ds_name]['mean_gradients'] = float(np.mean(all_grads, 0))

            stats[ds_name]['std_rgb'] = np2tup(np.std(all_rgbs, 0))
            stats[ds_name]['std_hsv'] = np2tup(np.std(all_hsvs, 0))
            stats[ds_name]['std_gradients'] = float(np.std(all_grads, 0))

        output['global_statistics'] = stats

    if mode.psd:
        fig = plt.figure()
        ax = plt.axes()
        ax.set_title('Average Power Spectral Density')
        ax.set_xlabel('frequency [pixels]')
        ax.set_ylabel('intensity [counts]')

        for ds_name in datasets.keys():
            try:
                depth_list = [x['depth_data'] for x in all_data[ds_name]]
                xf, psd = get_psd_of_depth_list(depth_list)
                ax.loglog(xf, psd, label=ds_name)
            except KeyError:
                pass

        plt.legend()
        plt.grid()
        plt.show()

    if len(output) > 0:
        with open("output.json", 'w') as F:
            pass
            # json.dump(output, F, indent=2)

    if args.vis:
        # extract images for visualization
        # for idx in range(len(inds)):
        ind_it = range(len(inds)).__iter__()

        def on_press(event):
            idx = ind_it.__next__()

            vis_data = {'rgb': [], 'depth': [], 'mask': [], 'normals': [], 'bboxes': [], 'pose': []}
            for ds_name in datasets.keys():
                for key in vis_data.keys():
                    try:
                        img = all_data[ds_name][idx][key]
                        if preview_size is not None:
                            img = cv2.resize(img, preview_size[::-1])
                        vis_data[key].append(img)

                    except KeyError:
                        pass
            img_id = 1
            for key, val in vis_data.items():
                if len(val) > 0:
                    val = np.concatenate(val, 1)
                    plt.subplot(3, 2, img_id, ymargin=0., xmargin=0.)
                    plt.title(key + ": " + " - ".join(datasets.keys()))
                    plt.imshow(val)
                    img_id += 1
                    # cv2.imshow(key + ": " + " - ".join(datasets.keys()), val)
            idx += 1
            if idx == len(inds):
                exit()
            fig.canvas.draw()

        fig = plt.figure()
        plt.subplot(3, 2, 1, ymargin=0., xmargin=0.)
        fig.canvas.mpl_connect('key_press_event', on_press)
        on_press(None)
        plt.show()


if __name__ == '__main__':
    main()
