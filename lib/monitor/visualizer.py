import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import colorsys
from lib.geometry.geometry import project_p3d
import tensorflow as tf
import io


def vis_3d_points(pts_1, pts_2=None, num_plot=2, save_path=None, axis_range=None):
    """
    visualize the points in 3D space
    :params pts_xyz: numpy array: [n_pts, 3]  n_pts: number of the points
            kpts_xyz (optional): numpy array: [n_kpts, 3] n_kpts: number of the key points

            axis_range: a list [axis_range_low, axis_range_high] e.g [-1, 1]
    """

    assert num_plot == 1 or 2, "Only supports one or two subplots"

    fig = plt.figure()
    ax = fig.add_subplot(1, num_plot, 1, projection='3d')
    for i in pts_1:
        xs = i[0]
        ys = i[1]
        zs = i[2]
        ax.scatter(xs, ys, zs, marker="o", color='b')

    if pts_2 is not None:
        if num_plot == 2:
            ax = fig.add_subplot(1, num_plot, num_plot, projection='3d')

        for k in pts_2:
            x_s = k[0]
            y_s = k[1]
            z_s = k[2]
            ax.scatter(x_s, y_s, z_s, marker="o", color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if axis_range is not None:
        ax.set_xlim3d(axis_range[0], axis_range[1])
        ax.set_ylim3d(axis_range[0], axis_range[1])
        ax.set_zlim3d(axis_range[0], axis_range[1])

    plt.show()

    if save_path is not None:
        plt.savefig(save_path)


def visualize_normal_map(normal_map):
    """ returns BGR image of normal map [h,w,3] according to color wheel """
    hue = np.arctan2(normal_map[:, :, 1], normal_map[:, :, 0]) / np.pi  # [-1, 1]
    hue += 1.0  # [0, 2]
    hue *= 90  # [0, 180
    sat = np.linalg.norm(normal_map[:, :, :2], axis=-1) * 255
    val = np.ones_like(hue) * 255
    hsv = np.stack([hue, sat, val], -1).astype(np.uint8)
    vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return vis


def draw_bbox(image, bboxes, cls_type='duck', show_label=True, show_confidence=True,
              Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    num_classes = 1 if cls_type != 'all' else 10

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5]) if cls_type == 'all' else 0
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]

        score_str = " {:.2f}".format(score) if show_confidence else ""
        if tracking: score_str = " " + str(score)
        try:
            label = "{}".format(cls_type) + score_str
        except KeyError:
            print("You received KeyError, this might be that you are trying to use yolo original weights")
            print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

        draw_single_bbox(image, coor, label, bbox_color, show_label=show_label)
    return image


def draw_single_bbox(img, bbox, label, color, show_label=True):
    image_h, image_w, _ = img.shape
    coor = np.array(bbox[:4], dtype=np.int32)
    bbox_thick = int(0.6 * (image_h + image_w) / 1000)
    if bbox_thick < 1: bbox_thick = 1
    fontScale = 0.75 * bbox_thick
    x1, y1, x2, y2 = coor
    # put object rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, bbox_thick * 2)

    if show_label:
        # get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                              fontScale, thickness=bbox_thick)
        # put filled text rectangle
        cv2.rectangle(img, (x1, y1), (x1 + text_width, y1 - text_height - baseline), color,
                      thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale, (255, 255, 255), bbox_thick, lineType=cv2.LINE_AA)


def project2img(mesh_pts, Rt, img, camera_intrinsic, camera_scale, color, xy_ofst):
    x1, y1 = xy_ofst
    mesh_pts_trans = np.dot(mesh_pts.copy(), Rt[:, :3].T) + Rt[:, 3]
    mesh_pt2ds = project_p3d(p3d=mesh_pts_trans, cam_scale=camera_scale, K=camera_intrinsic)
    mesh_pt2ds[:, 0] -= x1
    mesh_pt2ds[:, 1] -= y1
    img = draw_p2ds(img, p2ds=mesh_pt2ds, r=1, color=color) / 255.
    return img


def draw_p2ds(img, p2ds, r=10, color=(255, 0, 0)):
    h, w = img.shape[0], img.shape[1]
    for pt_2d in p2ds:
        pt_2d[0] = np.clip(pt_2d[0], 0, w)
        pt_2d[1] = np.clip(pt_2d[1], 0, h)
        img = cv2.circle(
            img, (pt_2d[0], pt_2d[1]), r, color, -1
        )
    return img


def dpt2heat(dpt, style='gray'):
    vd = dpt[dpt > 0]
    dpt[dpt > 0] = (dpt[dpt > 0] - vd.min()) / (vd.max() - vd.min() + 1e-6)
    colormap = plt.get_cmap(style)
    heatmap = (colormap(dpt.copy()) * 2 ** 16).astype(np.uint16)[:, :, :3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    return heatmap / 2 ** 16


def vis_pts_semantics(img, semantics, sampled_index):
    """
    :param img:  rgb image
    :param seg_pre:
    :param sampled_index:
    :return:
    """

    img_map_sample = np.zeros(shape=(img.shape[0] * img.shape[1]))
    img_map_seg = np.zeros(shape=(img.shape[0] * img.shape[1]))

    img_map_sample[sampled_index] = 1
    sampled_mask = img_map_sample.reshape((img.shape[0], img.shape[1]))
    sampled_mask_index = np.nonzero(sampled_mask)

    img_map_seg[sampled_index] = semantics
    seg_mask = img_map_seg.reshape((img.shape[0], img.shape[1]))
    seg_mask_index = np.nonzero(seg_mask)

    for x, y in zip(sampled_mask_index[1], sampled_mask_index[0]):
        img = cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    for x, y in zip(seg_mask_index[1], seg_mask_index[0]):
        img = cv2.circle(img, (x, y), 3, (128, 0, 128), -1)

    return img / 255.


def vis_offset_value(sampled_index, semantics, kpts_ofst, ctr_ofst, gt_pt2ds=None, pre_pt2ds=None,
                     img_shape=(480, 640, 3)):
    """
    Visilizing the offset value in heat map respectively for both gt and prediction
    :param sampled_index: indices for sampled pixels
    :param img_shape:
    :param kpts_ofst: ofst vectors from ground truth or prediction
    :param ctr_ofst:
    :return:
    """
    h, w, _ = img_shape
    offset_heat_map_list = []
    sampled_index = sampled_index[semantics == 1]
    kpts_cpts_offst = np.concatenate([kpts_ofst, ctr_ofst], axis=2).squeeze()  # [npts, 9, 3]
    kpts_cpts_offst = kpts_cpts_offst[semantics == 1]
    kpts_cpts_offst_perm = kpts_cpts_offst.transpose((1, 0, 2))  # [9, npts, 3]
    kpts_cpts_offst_norm = np.linalg.norm(kpts_cpts_offst_perm, ord=2, axis=-1)  # [9, npts]

    for i in range(kpts_cpts_offst_norm.shape[0]):
        try:  # catch the exception when the prediction is all wrong
            offset_norm = kpts_cpts_offst_norm[i]
            offset_map = np.ones(shape=(h * w))
            offset_map[tuple(sampled_index)] = offset_norm
            offset_map = np.reshape(offset_map, newshape=(h, w))
            heat_map = dpt2heat(offset_map) * 255
        except:
            heat_map = dpt2heat(np.ones(shape=(h, w))) * 255.

        # if len(offset_norm) == 0:  # here when len(semantics==1) == 0
        #     heat_map = dpt2heat(np.ones(shape=(h, w))) * 255
        # else:
        #     if offset_norm.max() == 0:  # here check if the predicted offset all zeros
        #         heat_map = dpt2heat(np.ones(shape=(h, w))) * 255
        #     else:
        #         heat_map = dpt2heat(offset_map) * 255

        if gt_pt2ds is not None:
            pt_2d = gt_pt2ds[i]
            heat_map = cv2.circle(heat_map, (pt_2d[0], pt_2d[1]), radius=4, color=(255, 0, 0)) / 255.

        if pre_pt2ds is not None:
            pre_pt = pre_pt2ds[i]
            heat_map = cv2.circle(heat_map, (pre_pt[0], pre_pt[1]), radius=4, color=(0, 0, 255)) / 255.

        offset_heat_map_list.append(heat_map)

    return offset_heat_map_list


def vis_gt_kpts(rgb, kpts, Rt, xy_offset, cam_matrix):
    color = (255, 0, 0)
    x1, y1 = xy_offset
    mesh_pts_trans = np.dot(kpts.copy(), Rt[:, :3].T) + Rt[:, 3]
    mesh_pt2ds = project_p3d(p3d=mesh_pts_trans, cam_scale=1.0, K=cam_matrix)
    mesh_pt2ds[:, 0] -= x1
    mesh_pt2ds[:, 1] -= y1
    rgb = draw_p2ds(rgb, p2ds=mesh_pt2ds, r=5, color=color) / 255.
    return rgb, mesh_pt2ds


def vis_pre_kpts(rgb, pre_kpts, xy_offset, cam_matrix):
    color = (0, 0, 255)
    x1, y1 = xy_offset
    mesh_pt2ds = project_p3d(p3d=pre_kpts, cam_scale=1.0, K=cam_matrix)
    mesh_pt2ds[:, 0] -= x1
    mesh_pt2ds[:, 1] -= y1
    rgb = draw_p2ds(rgb, p2ds=mesh_pt2ds, r=5, color=color) / 255.
    return rgb, mesh_pt2ds


def vis_feature_maps_image(feature_array):
    _, _, c = feature_array.shape
    col = 8
    row = int(c / col)
    ix = 1
    fig = plt.figure()
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_array[:, :, ix - 1], interpolation=None)
            ix += 1

    fig.set_size_inches(14, 9)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return tf.cast(image, tf.float32) / 255.


def vis_stochastic_poses(images):
    bs = len(images)
    col = 8
    row = int(bs / col)
    w = col * 1.75
    h = row * 1.75
    ix = 0
    fig = plt.figure()
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(images[ix], interpolation=None)
            ix += 1
    fig.set_size_inches(w, h)
    return fig


def vis_accuracy(distance_list, threshold, name):
    D = np.array(distance_list)
    D = np.sort(D)  # n
    n = len(distance_list)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n  # n

    # n_steps = 20
    # step_size = (max(D) - min(D)) / n_steps

    step_size = threshold
    max_dist = max(D)

    fig = plt.figure()
    weights = np.ones_like(D) / float(len(D))
    plt.hist(D, bins=np.arange(0., max_dist, step_size), weights=weights, label='binned samples')
    plt.plot(D, acc, label='ADD Score')
    plt.ylabel('Accuracy')
    plt.xlabel("Distance threshold")
    plt.xticks(np.arange(0., max_dist, step_size))
    # only show everye nth label (except threshold)
    ax = plt.gca()
    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
        label.set_visible((idx - 1) % 5 == 0)  # always show first==threshold

    plt.title("Accuracy - {} Metric".format(name))
    plt.vlines(x=threshold, ymin=0, ymax=1.0, linestyles='dashdot')
    plt.legend()

    plt.grid(True)
    fig.set_size_inches(14, 9)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return tf.cast(image, tf.float32) / 255.
