import os
import random
import numpy as np
import cv2
import tensorflow as tf
import pickle
import yaml
import math


def get_unity_depth_value(img_dpt, clip_range=(1.0, 2.5)):
    clip_range_l, clip_range_u = clip_range
    dpt_map = (img_dpt[:, :, 0] / 255) * (clip_range_u - clip_range_l) + clip_range_l
    return dpt_map


def load_mesh(ptxyz_pth, scale=0.001, n_points=2000):
    pointxyz = ply_vtx(ptxyz_pth) * scale
    dellist = [j for j in range(0, len(pointxyz))]
    dellist = random.sample(dellist, len(pointxyz) - n_points)
    pointxyz = np.delete(pointxyz, dellist, axis=0)
    return pointxyz


# def get_pointxyz_linemod(ptxyz_pth, scale=0.001):
#     pointxyz = ply_vtx(ptxyz_pth) * scale
#     dellist = [j for j in range(0, len(pointxyz))]
#     dellist = random.sample(dellist, len(pointxyz) - 2000)
#     pointxyz = np.delete(pointxyz, dellist, axis=0)
#     return pointxyz
#
#
# def get_pointxyz_blender(ptxyz_pth, scale=0.01):
#     pointxyz = ply_vtx(ptxyz_pth) * scale
#     dellist = [j for j in range(0, len(pointxyz))]
#     dellist = random.sample(dellist, len(pointxyz) - 2000)
#     pointxyz = np.delete(pointxyz, dellist, axis=0)
#     return pointxyz


def get_pointxyz_unity(cls_type_id):
    """
    return: point cloud in meters
    """
    ptxyz_pth = os.path.join(
        'dataset/unity/unity_obj_mesh/',
        'obj_%02d.ply' % cls_type_id
    )
    pointxyz = ply_vtx(ptxyz_pth) / 100
    pointxyz[:, 0] *= -1
    if len(pointxyz) < 2000:
        return pointxyz
    dellist = [j for j in range(0, len(pointxyz))]
    dellist = random.sample(dellist, len(pointxyz) - 2000)
    pointxyz = np.delete(pointxyz, dellist, axis=0)
    return pointxyz


def ply_vtx(pth):
    """read ply mesh file"""
    f = open(pth)
    assert f.readline().strip() == "ply"
    while True:
        line = f.readline().strip()
        if line.startswith("element vertex"):
            N = int(line.split()[-1])
            break
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))

    return np.array(pts)


@tf.function
def dpt_2_cld_tf(dpt, cam_scale, cam_intrinsic, xy_offset=(0, 0), depth_trunc=2.0):
    import tensorflow as tf
    """
    This function converts 2D depth image into 3D point cloud according to camera intrinsic matrix
    :param dpt: the 2d depth image
    :param cam_scale: scale converting units in meters
    :param cam_intrinsic: camera intrinsic matrix
    :param xy_offset: the crop left upper corner index on the original image

    P(X,Y,Z) = (inv(K) * p2d) * depth
    where:  P(X, Y, Z): the 3D points
            inv(K): the inverse matrix of camera intrinsic matrix
            p2d: the [ u, v, 1].T the pixels in the image
            depth: the pixel-wise depth value
     """

    h_depth = tf.shape(dpt)[0]
    w_depth = tf.shape(dpt)[1]

    y_map, x_map = tf.meshgrid(tf.range(w_depth, dtype=tf.float32),
                               tf.range(h_depth, dtype=tf.float32))  # vice versa than mgrid

    x_map = x_map + tf.cast(xy_offset[1], tf.float32)
    y_map = y_map + tf.cast(xy_offset[0], tf.float32)

    msk_dp = tf.math.logical_and(dpt > 1e-6, dpt < depth_trunc)
    msk_dp = tf.reshape(msk_dp, (-1,))

    pcld_index = tf.squeeze(tf.where(msk_dp))

    dpt_mskd = tf.expand_dims(tf.gather(tf.reshape(dpt, (-1,)), pcld_index), -1)
    xmap_mskd = tf.expand_dims(tf.gather(tf.reshape(x_map, (-1,)), pcld_index), -1)
    ymap_mskd = tf.expand_dims(tf.gather(tf.reshape(y_map, (-1,)), pcld_index), -1)

    pt2 = dpt_mskd / tf.cast(cam_scale, dpt_mskd.dtype)  # z
    cam_cx, cam_cy = cam_intrinsic[0][2], cam_intrinsic[1][2]
    cam_fx, cam_fy = cam_intrinsic[0][0], cam_intrinsic[1][1]

    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    pcld = tf.concat((pt0, pt1, pt2), axis=1)

    return pcld, pcld_index


def dpt_2_cld(dpt, cam_scale, cam_intrinsic, xy_offset=(0, 0), depth_trunc=2.0, downsample_factor=1):
    """
    This function converts 2D depth image into 3D point cloud according to camera intrinsic matrix
    :param dpt: the 2d depth image
    :param cam_scale: scale converting units in meters
    :param cam_intrinsic: camera intrinsic matrix
    :param xy_offset: the crop left upper corner index on the original image

    P(X,Y,Z) = (inv(K) * p2d) * depth
    where:  P(X, Y, Z): the 3D points
            inv(K): the inverse matrix of camera intrinsic matrix
            p2d: the [ u, v, 1].T the pixels in the image
            depth: the pixel-wise depth value
     """

    x1, y1 = xy_offset

    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]

    h_depth, w_depth = dpt.shape

    x_map, y_map = np.mgrid[:h_depth, :w_depth] * downsample_factor

    x_map += y1
    y_map += x1

    msk_dp = np.logical_and(dpt > 1e-6, dpt < depth_trunc)

    pcld_index = msk_dp.flatten().nonzero()[0].astype(np.uint32)  # index for nonzero elements

    if len(pcld_index) < 1:
        return None, None

    dpt_mskd = dpt.flatten()[pcld_index][:, np.newaxis].astype(np.float32)
    xmap_mskd = x_map.flatten()[pcld_index][:, np.newaxis].astype(np.float32)
    ymap_mskd = y_map.flatten()[pcld_index][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd / cam_scale  # z
    cam_cx, cam_cy = cam_intrinsic[0][2], cam_intrinsic[1][2]
    cam_fx, cam_fy = cam_intrinsic[0][0], cam_intrinsic[1][1]

    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    pcld = np.concatenate((pt0, pt1, pt2), axis=1)

    return pcld, pcld_index


def get_label_color(cls_id, n_cls=14):
    """
    assign color for each class in order of bgr
    """
    mul_col = 255 * 255 * 255 // n_cls * cls_id
    r, g, b = mul_col // 255 // 255, (mul_col // 255) % 255, mul_col % 255
    bgr = (int(r), int(g), int(b))
    return bgr


def get_offst(RT_list, pcld_xyz, mask_selected, n_objects, cls_type, centers, kpts, mask_value=255):
    """
        num_obj is == num_classes by default
    """
    RTs = np.zeros((n_objects, 3, 4))
    kp3ds = np.zeros((n_objects, 8, 3))
    ctr3ds = np.zeros((n_objects, 1, 3))
    kpts_targ_ofst = np.zeros((len(pcld_xyz), 8, 3))
    ctr_targ_ofst = np.zeros((len(pcld_xyz), 1, 3))

    for i in range(len(RT_list)):
        RTs[i] = RT_list[i]  # assign RT to each object
        r = RT_list[i][:3, :3]
        t = RT_list[i][:3, 3]

        if cls_type == 'all':
            pass
        else:
            cls_index = np.where(mask_selected == mask_value)[0]  # todo 255 for object

        ctr = centers[i][:, None]
        ctr = np.dot(ctr.T, r.T) + t  # ctr [1 , 3]
        ctr3ds[i] = ctr

        ctr_offset_all = []

        for c in ctr:
            ctr_offset_all.append(np.subtract(c, pcld_xyz))

        ctr_offset_all = np.array(ctr_offset_all).transpose((1, 0, 2))
        ctr_targ_ofst[cls_index, :, :] = ctr_offset_all[cls_index, :, :]

        kpts = kpts[i]
        kpts = np.dot(kpts, r.T) + t  # [8, 3]
        kp3ds[i] = kpts

        kpts_offset_all = []

        for kp in kpts:
            kpts_offset_all.append(np.subtract(kp, pcld_xyz))

        kpts_offset_all = np.array(kpts_offset_all).transpose((1, 0, 2))  # [kp, np, 3] -> [nsp, kp, 3]

        kpts_targ_ofst[cls_index, :, :] = kpts_offset_all[cls_index, :, :]

    return ctr_targ_ofst, kpts_targ_ofst


def get_pcld_rgb(rgb, pcld_index):
    """
    rgb : [h, w, c]
    return: pcld_rgb [h*w, c]
    """
    return rgb.reshape((-1, 3))[pcld_index]


def pcld_processor(depth, rgb, camera_matrix, camera_scale, n_points, xy_ofst=(0, 0), downsample_factor=1,
                   depth_trunc=2.0):
    """
    params: depth: depth_map from camera (or the cropped depth image)
            rgb: rgb data from camera (or the cropped rgb image)
    return:
            pcld_xyz
            pcld_feats
            sampled_index
    """

    pcld_xyz, pcld_index = dpt_2_cld(depth, camera_scale, camera_matrix, xy_ofst, depth_trunc=depth_trunc,
                                     downsample_factor=downsample_factor)
    if pcld_index is None:
        return None, None, None
    pcld_rgb = get_pcld_rgb(rgb, pcld_index)

    pcld_index_id = np.arange(len(pcld_index))

    if len(pcld_index_id) > n_points:
        c_mask = np.zeros(len(pcld_index_id), dtype=int)
        c_mask[:n_points] = 1
        np.random.shuffle(c_mask)
        pcld_index_id = pcld_index_id[c_mask.nonzero()]
    else:
        pcld_index_id = np.pad(pcld_index_id, (0, n_points - len(pcld_index_id)), "wrap")

    pcld_xyz_rgb = np.concatenate((pcld_xyz, pcld_rgb), axis=1)[pcld_index_id, :]

    pcld_nm = normalize_pcld_xyz(pcld_xyz[pcld_index_id, :])[:, :3]

    pcld_nm[np.isnan(pcld_nm)] = 0.0

    pcld_xyz_rgb_nm = np.concatenate((pcld_xyz_rgb, pcld_nm), axis=1)

    sampled_index = pcld_index[pcld_index_id]

    return pcld_xyz_rgb_nm[:, :3], pcld_xyz_rgb_nm[:, 3:], sampled_index


def pcld_processor_tf(depth, rgb, camera_matrix, camera_scale,
                      n_sample_points, xy_ofst=(0, 0), depth_trunc=2.0):
    points, valid_inds = dpt_2_cld_tf(depth, camera_scale, camera_matrix, xy_ofst, depth_trunc=depth_trunc)

    import tensorflow as tf

    n_valid_inds = tf.shape(valid_inds)[0]
    sampled_inds = tf.range(n_valid_inds)

    if n_valid_inds < 10:
        # because tf.function: return same dtypes
        return tf.constant([0.]), tf.constant([0.]), tf.constant([0], valid_inds.dtype)

    if n_valid_inds >= n_sample_points:
        sampled_inds = tf.random.shuffle(sampled_inds)

    else:
        repeats = tf.cast(tf.math.ceil(n_sample_points / n_valid_inds), tf.int32)
        sampled_inds = tf.tile(sampled_inds, [repeats])

    sampled_inds = sampled_inds[:n_sample_points]

    final_inds = tf.gather(valid_inds, sampled_inds)

    points = tf.gather(points, sampled_inds)

    rgbs = tf.reshape(rgb, (-1, 3))
    rgbs = tf.gather(rgbs, final_inds)

    normals = compute_normals(depth, camera_matrix)
    normals = tf.gather(normals, final_inds)

    feats = tf.concat([rgbs, normals], 1)
    return points, feats, final_inds

@tf.function
def compute_normal_map(depth, camera_matrix):
    kernel = np.array([[[[0.5, 0.5]], [[-0.5, 0.5]]], [[[0.5, -0.5]], [[-0.5, -0.5]]]])

    diff = tf.nn.conv2d(depth, kernel, 1, "VALID")

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    scale_depth = tf.concat([depth / fx, depth / fy], -1)

    # clip=tf.constant(1)
    # diff = tf.clip_by_value(diff, -clip, clip)
    diff = diff / scale_depth[:, :-1, :-1, :]  # allow nan -> filter later

    mask = tf.logical_and(~tf.math.is_nan(diff), tf.abs(diff) < 5)

    diff = tf.where(mask, diff, 0.)

    smooth = tf.constant(4)
    kernel2 = tf.cast(tf.tile([[1 / tf.pow(smooth, 2)]], (smooth, smooth)), tf.float32)
    kernel2 = tf.expand_dims(tf.expand_dims(kernel2, axis=-1), axis=-1)
    kernel2 = kernel2 * tf.eye(2, batch_shape=(1, 1))
    diff2 = tf.nn.conv2d(diff, kernel2, 1, "VALID")

    mask_conv = tf.nn.conv2d(tf.cast(mask, tf.float32)
                             , kernel2, 1, "VALID")

    diff2 = diff2 / mask_conv

    ones = tf.expand_dims(tf.ones(diff2.shape[:3]), -1)
    v_norm = tf.concat([diff2, ones], axis=-1)

    v_norm, _ = tf.linalg.normalize(v_norm, axis=-1)
    v_norm = tf.where(~tf.math.is_nan(v_norm), v_norm, [0])

    v_norm = - tf.image.resize_with_crop_or_pad(v_norm, depth.shape[1], depth.shape[2])  # pad and flip (towards cam)
    return v_norm


def compute_normals(depth, camera_matrix):
    import tensorflow as tf
    depth = tf.expand_dims(tf.expand_dims(depth, axis=-1), axis=0)
    normal_map = compute_normal_map(depth, camera_matrix)
    normals = tf.reshape(normal_map[0], (-1, 3))  # reshape als list of normals
    return normals


def rescale_image_bbox(image, target_size, gt_boxes=None, interpolation=cv2.INTER_LINEAR):
    """
    rescale an image to target_size, correspondingly rescale gt_boxes
    params:
        target_size: a list or tuple (target_h, target_w)
    """
    ih, iw = target_size
    h, w = image.shape[:2]
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)

    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image_resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
    if len(image.shape) == 3:
        image_paded = np.full(shape=[ih, iw, 3], fill_value=0.)
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized

    else:
        image_paded = np.full(shape=[ih, iw], fill_value=0.)
        image_paded[dh:nh + dh, dw:nw + dw] = image_resized

    if gt_boxes is None:
        return image_paded

    else:
        if len(gt_boxes)>0:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes
        else:
            return image_paded, np.array([])


def crop_image(img, crop_index):
    """ crop image [H,W...] to bounding box [x1, y1, x2, y2] """
    x1, y1, x2, y2 = crop_index
    img = img[y1:y2, x1:x2]
    return img


def get_bbox_from_mask(mask, gt_mask_value=255):
    """ mask with object as 255 -> bbox [x1,y1, x2, y2]"""

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    y, x = np.where(mask == gt_mask_value)
    inds = np.stack([x, y])
    if 0 in inds.shape:
        return None
    x1, y1 = np.min(inds, 1)
    x2, y2 = np.max(inds, 1)

    return (x1, y1, x2, y2)


def get_bbox(p3d, cam_scale, K, rgb_shape=(480, 640)):
    """ calculate bbox [x1, y1, x2, y2] from projecting points onto image """
    h, w = rgb_shape
    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    x_min = np.min(p2d[:, 0]).clip(min=0, max=w)
    x_max = np.max(p2d[:, 0]).clip(min=0, max=w)
    y_min = np.min(p2d[:, 1]).clip(min=0, max=h)
    y_max = np.max(p2d[:, 1]).clip(min=0, max=h)
    bbox = np.array([x_min, y_min, x_max, y_max])
    return bbox.tolist()


def validate_bbox(bbox):
    x1, y1, x2, y2 = bbox[:4]
    size = (y2 - y1) * (x2 - x1)
    return False if size < 25 else True


def get_crop_index(bbox, rgb_size=(480, 640), base_crop_resolution=(160, 160)):
    """
    get the crop index on original images according to the predicted bounding box
    params: bbox: predicted bbox
            rgb_size: height and width of the original input image
            target_resolution: the resolution of the target cropped images
    return:
            crop_index(left upper corner and right bottom corner) on the original image
    """
    ori_img_h, ori_img_w = rgb_size[:2]

    coor = np.array(bbox[:4], dtype=np.int32)
    (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

    x_c = (x1 + x2) * 0.5
    y_c = (y1 + y2) * 0.5
    h_t, w_t = base_crop_resolution

    # size_bbox = (y2 - y1) * (x2 - x1)
    # size_base_crop = w_t * h_t
    # if size_bbox >= 2 * size_base_crop:
    # crop_factor = 3
    # elif size_bbox >= size_base_crop:
    # crop_factor = 2
    # else:
    # crop_factor = 1

    bbox_w, bbox_h = (x2 - x1), (y2 - y1)
    w_factor = bbox_w / base_crop_resolution[0]
    h_factor = bbox_h / base_crop_resolution[1]
    crop_factor = math.ceil(max(w_factor, h_factor))

    w_t *= crop_factor
    h_t *= crop_factor

    x1_new = x_c - w_t * 0.5
    x2_new = x_c + w_t * 0.5
    y1_new = y_c - h_t * 0.5
    y2_new = y_c + h_t * 0.5

    if x1_new < 0:
        x1_new = 0
        x2_new = x1_new + w_t

    if x2_new > ori_img_w:
        x2_new = ori_img_w
        x1_new = x2_new - w_t

    if y1_new < 0:
        y1_new = 0
        y2_new = y1_new + h_t

    if y2_new > ori_img_h:
        y2_new = ori_img_h
        y1_new = y2_new - h_t

    return np.array([x1_new, y1_new, x2_new, y2_new]).astype(dtype=np.int), crop_factor


def convert2pascal(box_coor):
    """
    params: box_coor [xmin, ymin, w, h]
    return: box_coor in pascal_voc format [xmin, ymin, xmax, ymax] (left upper corner and right bottom corner)
    """
    x_min, y_min, w, h = box_coor
    x_max = x_min + w
    y_max = y_min + h
    return [x_min, y_min, x_max, y_max]


def tf_normalize_image(rgb):
    import tensorflow as tf
    average = tf.reduce_mean(rgb, axis=(-3, -2), keepdims=True)
    min = tf.reduce_min(rgb, axis=(-3, -2), keepdims=True)
    max = tf.reduce_max(rgb, axis=(-3, -2), keepdims=True)
    rgb = (rgb - average) / (max - min)
    return rgb


def get_data_preprocessed_linemod(preprocessed_data_path):
    preprocessed_data = pickle.load(open(preprocessed_data_path, "rb"))
    rgb = preprocessed_data['rgb']
    # crop_info = preprocessed_data['crop_info']
    rt = preprocessed_data['RT']
    if 'depth' in preprocessed_data.keys():
        depth = preprocessed_data['depth']
    else:
        depth = None
    cam_intrinsic = preprocessed_data['K']
    # crop_index, crop_down_factor = crop_info
    return rt, rgb, depth, cam_intrinsic


def get_data_preprocessed_generic(path, index, names):
    get_data = lambda name: np.load(os.path.join(path, name, f"{index:06}.npy"))
    return (get_data(name) for name in names)


def expand_dim(*argv):
    item_lst = []
    for item in argv:
        item = np.expand_dims(item, axis=0)
        item_lst.append(item)
    return item_lst


def get_bbox_pascal(x_c, y_c, width, height):
    x_min = x_c
    x_max = x_c + width
    y_min = y_c
    y_max = y_c + height
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(int)
    return bbox.tolist()


def get_mesh_diameter(mesh_info_path, obj_id):
    with open(mesh_info_path, "r") as mesh_info:
        mesh_info = yaml.load(mesh_info, Loader=yaml.FullLoader)
        diameter = mesh_info[obj_id]["diameter"]
    return diameter


def formatting_predictions(box_best, yolo_rescale_factor, dw, dh):
    label, confidence, bbox = box_best
    x, y, w, h = bbox

    xmin =int((x - w / 2 - dw) / yolo_rescale_factor)
    xmax = int((x + w / 2 - dw) / yolo_rescale_factor)
    ymin = int((y - h / 2 - dh) / yolo_rescale_factor)
    ymax = int((y + h / 2 - dh) / yolo_rescale_factor)
    bbox = [xmin, ymin, xmax, ymax, float(confidence)]

    return bbox


def get_yolo_rescale_values(input_image_size=(480, 640), target_image_size=(416, 416)):
    ih, iw = target_image_size
    h, w = input_image_size
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    return scale, dw, dh
