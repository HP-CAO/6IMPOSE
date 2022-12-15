import math

import numpy
import tensorflow as tf
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
from sklearn.cluster import MeanShift


# import tensorflow_probability as tfp


def cal_cam_K(img_w, img_h, camera_h_fov=24.28699, camera_v_hov=14.1046):
    """
    calculate the camera's intrinsic matrix
    params: img_w: image_width
            img_h: image_y
            camera_h_hov: horizontal field of view in degrees
    return: camera_intrinsic matrix
    """

    f_x = img_w / (2 * math.tan(camera_h_fov * math.pi / 360))
    f_y = img_h / (2 * math.tan(camera_v_hov * math.pi / 360))
    cam_x = img_w / 2
    cam_y = img_h / 2
    cam_K = [[f_x, 0, cam_x],
             [0, f_y, cam_y],
             [0, 0, 1]]

    return cam_K


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    """

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T


def filter_offset_outliers(offset_pre, top_n=2000):
    """
    Filtering out the outliers by removing the prediction with larger distance
    :param offset_pre: [n_kp_cp, n_pts, 3]
    :param top_n: set threshold for remaining the top_n points with lower distance
    :return: offset_top_n_index [n_kp_cp, n_pts, 3]
    """
    offset_pre_norm = -1 * tf.math.reduce_euclidean_norm(offset_pre, axis=-1)
    offset_top_n_index = tf.math.top_k(offset_pre_norm, k=top_n)
    return offset_top_n_index


def pts_clustering_with_std(pts, sigma=tf.constant(1.)):
    """
    filtering the points with the standard derivation without batch
    :param sigma: 3 sigma to filtering the outliers
    :param pts: kps-candidates [n_kp_cp, n_pts, 3]
    :return: return the voted kps [n_kp_cp, 3] by averaging the all key points filtered
    """

    pts_transpose = tf.transpose(pts, perm=(0, 2, 1))  # [9, 3, n_pts]
    pts_std_xyz = tf.math.reduce_std(pts_transpose, axis=-1, keepdims=True)  # [n_kp_cp, 3, 1] the std for x,y,z channel
    pts_mean_xyz = tf.math.reduce_mean(pts_transpose, axis=-1,
                                       keepdims=True)  # [n_kp_cp, 3, 1] the mean for x,y, z channel
    filter_threshold = tf.multiply(pts_std_xyz, sigma)  # [n_kp_cp, 3, 1]
    pts_mask = tf.math.abs(pts_transpose - pts_mean_xyz) < filter_threshold  # [n_kp_cp, 3, n_pts]
    pts_mask = tf.cast(pts_mask, dtype=pts_transpose.dtype)
    pts_masked = tf.multiply(pts_transpose, pts_mask)  # the reason of using multiply is to maintain the dimensions
    pts_masked_non_zeros = tf.math.count_nonzero(pts_masked, axis=2, dtype=pts_transpose.dtype)  # [n_kp_cp, 3, 1]
    pts_sum = tf.math.reduce_sum(pts_masked, axis=2)
    filtered_mean_xyz = pts_sum / (pts_masked_non_zeros + 1e-3)  # [n_kp_cp, 3, 1]
    return filtered_mean_xyz, pts_mask


@tf.function
def batch_pts_clustering_with_std(kps_cans, sigma=1):
    """
    filtering the points with the standard derivation in batch
    :param sigma: 3 sigma to filtering the outliers
    :param kps_cans: kps-candidates [bs, n_kp_cp, n_pts, 3]
    :return: the voted kps [bs, n_kp_cp, 3] by averaging the all key points filtered
    """

    kps_cans_transpose = tf.transpose(kps_cans, perm=(0, 1, 3, 2))  # [bs, n_kp_cp, 3, n_pts]
    std = tf.math.reduce_std(kps_cans_transpose, axis=-1, keepdims=True)  # std for x y z channels [bs, n_kp_cp, 3, 1]
    mean = tf.math.reduce_mean(kps_cans_transpose, axis=-1,
                               keepdims=True)  # mean for x y z channels [bs, n_kp_cp, 3, 1]
    threshold = tf.multiply(std, sigma)  # [bs, n_kp_cp, 3, 1]
    kps_mask = tf.math.abs(kps_cans_transpose - mean) < threshold  # [bs, n_kp_cp, 3, 1]
    kps_mask = tf.cast(kps_mask, dtype=tf.float32)
    kpts_filtered = tf.multiply(kps_cans_transpose, kps_mask)
    non_zeros = tf.math.count_nonzero(kpts_filtered, axis=3, dtype=kps_cans.dtype)
    new_mean = tf.math.reduce_sum(kpts_filtered, axis=3) / non_zeros  # [bs, n_kp_cp_, 3]
    return new_mean


@tf.function
def rt_svd_transform(A, B):
    """
    Calculates the svd transform that maps corresponding points A to B in m spatial dimensions not in batch
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl, dim by default: [9, 3]
        B: Nxm numpy array of corresponding points, usually points on camera axis, dim by default: [9, 3]
        centroid_A: provided centroid for partial icp
        centroid_B: provided centroid for partial icp
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    """

    n_kp_cp, _ = A.shape

    A_points_trans = tf.transpose(A, perm=(1, 0))  # [3, 9]
    B_points_trans = tf.transpose(B, perm=(1, 0))  # [3, 9]

    # forming the weights matrix 9x9

    weights_vector = tf.ones(n_kp_cp, dtype=A.dtype)
    weights_matrix = tf.linalg.diag(weights_vector)  # [9, 9]

    weighted_centroid_A = tf.reduce_mean(tf.matmul(A_points_trans, weights_matrix), axis=1, keepdims=True)  # [3, 1]
    weighted_centroid_B = tf.reduce_mean(tf.matmul(B_points_trans, weights_matrix), axis=1, keepdims=True)  # [3, 1]

    center_vector_A = A_points_trans - weighted_centroid_A  # [3, 9]
    center_vector_B = B_points_trans - weighted_centroid_B  # [3, 9]

    covariance_matrix = \
        tf.matmul(tf.matmul(center_vector_A, weights_matrix), tf.transpose(center_vector_B, perm=(1, 0)))

    S, U, V = tf.linalg.svd(
        covariance_matrix)  # decomposition, where dim of s is [1, 9] and u,v [3, 9], svd does not support HALF
    det_v_ut = tf.linalg.det(tf.matmul(V, tf.transpose(U, perm=(1, 0))))  # also does not support HALF

    ones_vector = tf.ones(shape=(1, 3), dtype=A.dtype) if det_v_ut > 0 else tf.constant([1., 1., -1.], dtype=A.dtype)

    mid_matrix = tf.linalg.diag(ones_vector)  # [3, 3]
    R = tf.matmul(tf.matmul(V, mid_matrix), tf.transpose(U, perm=(1, 0)))
    t = weighted_centroid_B - tf.matmul(R, weighted_centroid_A)

    R = tf.squeeze(R)
    t = tf.squeeze(t)

    return R, t


@tf.function
def rt_svd_transform_fast(A, B):
    """
    Calculates the svd transform that maps corresponding points A to B in m spatial dimensions not in batch
    In this implementation, we don't calculate the importance weights
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl, dim by default: [9, 3]
        B: Nxm numpy array of corresponding points, usually points on camera axis, dim by default: [9, 3]
        centroid_A: provided centroid for partial icp
        centroid_B: provided centroid for partial icp
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    """

    n_kp_cp, _ = A.shape

    A_points_trans = tf.transpose(A, perm=(1, 0))  # [3, 9]
    B_points_trans = tf.transpose(B, perm=(1, 0))  # [3, 9]

    centroid_a = tf.reduce_mean(A_points_trans, axis=1, keepdims=True)  # [3, 1]
    centroid_b = tf.reduce_mean(B_points_trans, axis=1, keepdims=True)  # [3, 1]

    center_vector_A = A_points_trans - centroid_a  # [3, 9]
    center_vector_B = B_points_trans - centroid_b  # [3, 9]

    covariance_matrix = tf.matmul(center_vector_A, tf.transpose(center_vector_B, perm=(1, 0)))

    S, U, V = tf.linalg.svd(
        covariance_matrix)  # decomposition, where dim of s is [1, 9] and u,v [3, 9], svd does not support HALF
    det_v_ut = tf.linalg.det(tf.matmul(V, tf.transpose(U, perm=(1, 0))))  # also does not support HALF

    ones_vector = tf.ones(shape=(1, 3), dtype=A.dtype) if det_v_ut > 0 else tf.constant([1., 1., -1.], dtype=A.dtype)

    mid_matrix = tf.linalg.diag(ones_vector)  # [3, 3]
    R = tf.matmul(tf.matmul(V, mid_matrix), tf.transpose(U, perm=(1, 0)))
    t = centroid_b - tf.matmul(R, centroid_a)

    R = tf.squeeze(R)
    t = tf.squeeze(t)

    return R, t


@tf.function
def batch_rt_svd_transform(A, B, weights_vector):
    """
    Calculates the svd transform that maps corresponding points A to B in m spatial dimensions in batch
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl, dim by default: [bs, 9, 3]
        B: Nxm numpy array of corresponding points, usually points on camera axis, dim by default: [bs, 9, 3]
        centroid_A: provided centroid for partial icp
        centroid_B: provided centroid for partial icp
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    """
    # print("A_1st", A[0, :4, :])
    # print("B_1st", B[0, :4, :])

    bs, n_pts, _ = A.shape
    A_points_trans = tf.transpose(A, perm=(0, 2, 1))  # [bs, 3, n_pts]
    B_points_trans = tf.transpose(B, perm=(0, 2, 1))  # [bs, 3, n_pts]

    weights_matrix = tf.linalg.diag(weights_vector)

    num_non_zeros = tf.math.count_nonzero(weights_vector, dtype=tf.float32)

    weighted_A = tf.matmul(A_points_trans, weights_matrix)

    # print("weighted_A:", weighted_A)

    weighted_B = tf.matmul(B_points_trans, weights_matrix)

    weighted_centroid_A = \
        tf.reduce_sum(weighted_A, axis=2, keepdims=True) / num_non_zeros  # [bs, 3, 1]

    # print("weighted_centroid_A:", weighted_centroid_A)

    weighted_centroid_B = \
        tf.reduce_sum(weighted_B, axis=2, keepdims=True) / num_non_zeros  # [bs, 3, 1]

    center_vector_A = A_points_trans - weighted_centroid_A  # [bs, 3, n_pts]
    center_vector_B = B_points_trans - weighted_centroid_B  # [bs, 3, n_pts]

    covariance_matrix = \
        tf.matmul(tf.matmul(center_vector_A, weights_matrix),
                  tf.transpose(center_vector_B, perm=(0, 2, 1)))  # [bs, n_pts, 3]

    # print("covariance_matrix:", covariance_matrix)

    S, U, V = tf.linalg.svd(covariance_matrix)
    det_v_ut = tf.linalg.det(tf.matmul(V, tf.transpose(U, perm=(0, 2, 1))))  # also does not support HALF bs

    det_signs = tf.sign(det_v_ut)
    ones_vector = tf.ones(shape=(bs, 1)) * tf.expand_dims(det_signs, axis=1)
    ones_vector = tf.concat([tf.ones(shape=(bs, 2)), ones_vector], axis=1)

    mid_matrix = tf.linalg.diag(ones_vector)
    R = tf.matmul(tf.matmul(V, mid_matrix), tf.transpose(U, perm=(0, 2, 1)))
    t = weighted_centroid_B - tf.matmul(R, weighted_centroid_A)

    # print("A2B_t", t[0])
    return R, t


def project_p3d(p3d, cam_scale, K):
    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)  # this is in pixel
    return p2d


def invert_se3_matrix_unity(RT):
    last_row = np.array([[0, 0, 0, 1]]).astype(np.float32)
    R_inv = RT[:3, :3].T
    RT_inv = np.concatenate([R_inv, - np.dot(R_inv, RT[:3, 3:])], axis=1)
    RT_inv = np.concatenate([RT_inv, last_row], axis=0)
    return RT_inv


def matrix_from_6vector_euler_unity(obj_p, obj_r, degrees=True, euler="zxy", euler_order=(2, 0, 1)):
    obj_r = np.take(obj_r, euler_order).astype(np.float32)
    last_row = np.array([[0, 0, 0, 1]]).astype(np.float32)
    R_O_W = Rotation.from_euler(euler, obj_r, degrees=degrees)
    RT_O_W = np.concatenate([R_O_W.as_matrix(), np.array([[x, ] for x in obj_p])], axis=1)
    RT_O_W = np.concatenate([RT_O_W, last_row], axis=0)
    return RT_O_W


def matrix_from_6vector_quat_unity(obj_p, quat=None):
    last_row = np.array([[0, 0, 0, 1]]).astype(np.float32)
    R_O_W = Rotation.from_quat(quat)
    RT_O_W = np.concatenate([R_O_W.as_matrix(), np.array([[x, ] for x in obj_p])], axis=1)
    RT_O_W = np.concatenate([RT_O_W, last_row], axis=0)
    return RT_O_W


def get_Rt_unity(cam_r, cam_p, obj_r, obj_p):
    y_mirroring = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
    RT_O_W = matrix_from_6vector_quat_unity(obj_p, obj_r)
    RT_C_W = matrix_from_6vector_quat_unity(cam_p, cam_r)
    RT_O_C = y_mirroring @ invert_se3_matrix_unity(RT_C_W) @ RT_O_W
    return RT_O_C


@tf.function
def get_pt_candidates_tf(pcld_xyz, kpts_ofst_pre, seg_pre, ctr_ofst_pre, ratio=tf.constant(0.2)):
    """
    Applying segmentation filtering and outlier filtering
    input are batches over the same image
    :param pcld_xyz: point cloud input
    :param kpts_ofst_pre: key point offset prediction from pvn3d
    :param seg_pre: segmentation prediction from pvn3d
    :param ctr_ofst_pre: center point prediction from pvn3d
    :param ratio: the ratio of remaining points (lower norm distance)
    :return: the predicted and clustered key-points [batch, 9, 3]
    """
    seg = tf.argmax(seg_pre, axis=-1)  # [npts]
    kpts_cpts_offst_pre = tf.concat([kpts_ofst_pre, ctr_ofst_pre], axis=1)  # [npts, 9, 3]

    obj_pts_index = tf.where(seg == 1)[:, 0]  # get "row" indices

    kpts_cpts_offst_pre = tf.gather(kpts_cpts_offst_pre, obj_pts_index)
    kpts_cpts_offst_pre_perm = tf.transpose(kpts_cpts_offst_pre, (1, 0, 2))  # [9, npts_seg, 3]

    len_obj_pts_index = tf.shape(obj_pts_index)[0]
    kpts_cpts_offst_pre_norm = tf.norm(kpts_cpts_offst_pre_perm, axis=-1)  # [ 9, npts_seg]

    num_bottom = tf.cast(ratio * tf.cast(len_obj_pts_index, tf.float32), tf.int32) - 1
    kpts_cpts_index = tf.argsort(kpts_cpts_offst_pre_norm, axis=-1)  # ascending [9, npts]
    top_kpts_cpts_index = kpts_cpts_index[:, :num_bottom]  # [9, ratio*npts]

    kpts_cpts_offst_pre_filtered = tf.gather(kpts_cpts_offst_pre_perm, top_kpts_cpts_index,
                                             batch_dims=1)  # [9, ratio*npts, 3]

    n_kpts_cpts = tf.shape(kpts_cpts_offst_pre)[1]

    pcld_xyz = tf.gather(pcld_xyz, obj_pts_index, )  # [n_pts_seg, 3] # segmented
    pcld_xyz = tf.expand_dims(pcld_xyz, 0)
    pcld_xyz = tf.repeat(pcld_xyz, n_kpts_cpts, 0)  # [9, n_pts_seg, 3]
    pcld_xyz = tf.gather(pcld_xyz, top_kpts_cpts_index, batch_dims=1)

    kpts_cpts_can_ = pcld_xyz + kpts_cpts_offst_pre_filtered  # [9, n_pts_seg, 3]
    # we return filtered pcld_xyz for icp
    return kpts_cpts_can_


@tf.function
def batch_get_pt_candidates_tf(pcld_xyz, kpts_ofst_pre, seg_pre, ctr_ofst_pre, k=10):
    # currently k=10 is working pretty good
    """
    Applying segmentation filtering and outlier filtering
    input are batches over the same image
    :param pcld_xyz: point cloud input
    :param kpts_ofst_pre: key point offset prediction from pvn3d
    :param seg_pre: segmentation prediction from pvn3d
    :param ctr_ofst_pre: center point prediction from pvn3d
    :param ratio: the ratio of remaining points (lower norm distance)
    :return: the predicted and clustered key-points [batch, 9, 3]
    """
    k = tf.constant(k, dtype=tf.int32)
    kpts_cpts_offst_pre = tf.concat([kpts_ofst_pre, ctr_ofst_pre], axis=2)  # [bs, n_pts, n_kp_cp, 3]
    kpts_cpts_offst_pre_perm = tf.transpose(kpts_cpts_offst_pre, perm=(0, 2, 1, 3))  # [bs, n_kp_cp, n_pts, 3 ]

    bs, n_kp_cp, n_pts, c = kpts_cpts_offst_pre_perm.shape
    seg = tf.argmax(seg_pre, axis=-1)  # [bs, n_pts]
    seg = tf.repeat(tf.expand_dims(seg, axis=1), axis=1, repeats=n_kp_cp)
    seg = tf.repeat(tf.expand_dims(seg, axis=-1), axis=-1, repeats=c)
    seg_inv = tf.cast(tf.ones_like(seg) - seg, dtype=tf.float32)
    seg_inf = seg_inv * tf.constant(1000., dtype=tf.float32)  # [bs, n_kp_cp, n_pts]

    kpts_cpts_offst_inf = kpts_cpts_offst_pre_perm + seg_inf  # background points will have large distance
    kpts_cpts_offst_pre_perm_norm = tf.linalg.norm(kpts_cpts_offst_inf, axis=-1)  # [bs, n_kp_cp, n_pts]
    _, indices = tf.math.top_k(-1 * kpts_cpts_offst_pre_perm_norm, k=k)  # [bs, n_kp_cp, k]
    offst_selected = tf.gather(kpts_cpts_offst_pre_perm, indices, batch_dims=2)  # [bs, n_kp_cp, k, c]
    pcld_repeats = tf.repeat(tf.expand_dims(pcld_xyz, axis=1), axis=1, repeats=n_kp_cp)  # [bs, n_kp_cp, n_pts, c]
    pcld_repeats_selected = tf.gather(pcld_repeats, indices, batch_dims=2)

    kpts_cpts_can = pcld_repeats_selected + offst_selected
    return kpts_cpts_can


def get_pt_candidates(pcld_xyz, kpts_ofst_pre, seg_pre, ctr_ofst_pre, ratio=0.2):
    """
    Applying segmentation filtering and  norm_distance based outlier filtering
    :param pcld_xyz: point cloud input
    :param kpts_ofst_pre: key point offset prediction from pvn3d
    :param seg_pre: segmentation prediction from pvn3d
    :param ctr_ofst_pre: center point prediction from pvn3d
    :param ratio: the ratio of remaining points (lower norm distance)
    :return: the predicted key-points [9, n_points, 3]
    """
    assert kpts_ofst_pre.shape[0] == 1, "Batch_size != 1"

    if type(pcld_xyz) is not np.ndarray:
        pcld_xyz = np.array(pcld_xyz)

    segs = np.argmax(seg_pre.numpy(), axis=-1).squeeze()  # kill the batch dim
    obj_pts_index = np.where(segs == 1)[0]

    len_obj_pts_index = obj_pts_index.shape
    pcld_xyz = np.expand_dims(pcld_xyz, axis=1)

    # get points candidates
    kpts_cpts_offst_pre = np.concatenate([kpts_ofst_pre.numpy(), ctr_ofst_pre.numpy()],
                                         axis=2).squeeze()  # [npts, 9, 3]
    kpts_cpts_offst_pre = kpts_cpts_offst_pre[obj_pts_index]
    kpts_cpts_offst_pre_perm = kpts_cpts_offst_pre.transpose((1, 0, 2))  # [9, npts_seg, 3]
    kpts_cpts_offst_pre_norm = np.linalg.norm(kpts_cpts_offst_pre_perm, ord=2, axis=-1)  # [9, npts_seg]
    num_bottom = numpy.multiply(ratio, len_obj_pts_index).astype(np.int)[
                     0] - 1  # filtering out the 20% outliers with higher norm distance

    kpts_cpts_index = np.expand_dims(np.argpartition(kpts_cpts_offst_pre_norm, num_bottom)[:, :num_bottom], axis=-1)
    kpts_cpts_offst_pre_filtered = np.take_along_axis(kpts_cpts_offst_pre_perm, kpts_cpts_index, axis=1)

    _, n_kpts_cpts, _ = kpts_cpts_offst_pre.shape
    pcld_xyz = pcld_xyz[obj_pts_index]  # [npts_seg, 1, 3]
    pcld_reps = np.repeat(pcld_xyz, repeats=n_kpts_cpts, axis=1)  # [npts_seg, 9, 3]
    pcld_reps = pcld_reps.transpose((1, 0, 2))  # [9, npts_seg, 3]
    pcld_reps = np.take_along_axis(pcld_reps, kpts_cpts_index, axis=1)

    kpts_cpts_can = np.add(pcld_reps, kpts_cpts_offst_pre_filtered)  # [9, n_filtered_npts, 3]

    return kpts_cpts_can, pcld_xyz


def pts_clustering(obj_kpts, bandwidth=None):  # todo reimplement it in tensorflow
    if bandwidth is not None:
        kpts_voted = [MeanShift(bandwidth=bandwidth, n_jobs=16).fit(kpt).cluster_centers_[0] for kpt in obj_kpts]
    else:
        kpts_voted = [np.mean(kpt, axis=0) for kpt in obj_kpts]

    return kpts_voted


def rt_linear_fit(mesh_kpts, kpts_voted):  # todo reimplement it in tensorflow
    Rt = best_fit_transform(mesh_kpts, kpts_voted)
    return Rt


def rotate_datapoint(img, mask, gt, rotation=None):
    if rotation is None:
        rotation = np.random.uniform(0, 360)

    (h, w) = img.shape[:2]
    center = (h // 2, w // 2)
    M = cv2.getRotationMatrix2D((center[1], center[0]), rotation, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))

    rot_M = Rotation.from_euler('xyz', [0, 0, rotation], degrees=True).as_matrix()
    cam_quat = gt['cam_rotation']
    cam_rot = Rotation.from_quat([*cam_quat[1:], cam_quat[0]])

    new_rot = cam_rot.as_matrix() @ rot_M
    new_quat = Rotation.from_matrix(new_rot).as_quat()
    gt['cam_rotation'] = [new_quat[-1], *new_quat[:3], ]
    return img, mask, gt


def icp_refinement(initial_pose, source, target, distance_threshold):
    import open3d
    source_pcld = open3d.geometry.PointCloud()
    source_pcld.points = open3d.utility.Vector3dVector(source)

    target_pcld = open3d.geometry.PointCloud()
    target_pcld.points = open3d.utility.Vector3dVector(target)

    target_pcld.voxel_down_sample(voxel_size=0.05)
    target_pcld.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    initial_pose = np.concatenate((initial_pose, [[0, 0, 0, 1]]), axis=0)

    result_icp = open3d.pipelines.registration.registration_icp(source_pcld, target_pcld, distance_threshold,
                                                                initial_pose,
                                                                open3d.pipelines.registration.TransformationEstimationPointToPlane())
    Rt_refined = result_icp.transformation

    return Rt_refined[:3]


@tf.function
def one_step_svd(R, t, A, B):
    n_xyz_pcld, c = B.shape
    n_xyz_mesh, c = A.shape

    A = tf.matmul(A, tf.transpose(R, perm=(1, 0))) + t  # [N, 3]
    B_repeats = tf.repeat(tf.expand_dims(B, axis=1), repeats=n_xyz_mesh, axis=1)  # [n_xyz_pcld, n_xyz_mesh, 3]
    A_repeats = tf.repeat(tf.expand_dims(A, axis=0), repeats=n_xyz_pcld, axis=0)  # [n_xyz_pcld, n_xyz_mesh, 3]
    distance_matrix = tf.linalg.norm(tf.subtract(B_repeats, A_repeats), axis=-1)  # [n_xyz_pcld, n_xyz_mesh]
    corr_index = tf.argmin(distance_matrix, axis=-1)  # [n_xyz_pcld]
    corr_mesh = tf.gather(A, indices=corr_index)  # [n_xyz_pcld, 3]
    # R_svd, t_svd = rt_svd_transform_fast(A=corr_mesh, B=B) # it's running as fast as before, sometimes slower
    R_svd, t_svd = rt_svd_transform(A=corr_mesh, B=B)
    return R_svd, t_svd


@tf.function
def batch_one_step_svd(R, t, A, B, weights_vector):
    """
    :param R: a batch of rotation, bs, 3, 3
    :param t: a batch of translation bs, 3, 1
    :param A: mesh points
    :param B: pcld points
    :return:
    """
    bs, _, _ = R.shape
    n_xyz_pcld, c = B.shape
    n_xyz_mesh, c = A.shape
    t = tf.repeat(tf.expand_dims(t, 1), repeats=n_xyz_mesh, axis=1)  # bs, N, 3
    A = tf.matmul(A, tf.transpose(R, perm=(0, 2, 1))) + t  # bs, N, 3
    B_repeats = tf.repeat(tf.expand_dims(B, axis=1), repeats=n_xyz_mesh, axis=1)  # [n_xyz_pcld, n_xyz_mesh, 3]
    A_repeats = tf.repeat(tf.expand_dims(A, axis=1), repeats=n_xyz_pcld, axis=1)  # [bs, n_xyz_pcld, n_xyz_mesh, 3]
    distance_matrix = tf.linalg.norm(tf.subtract(A_repeats, B_repeats), axis=-1)  # [bs, n_xyz_pcld, n_xyz_mesh]
    corr_index = tf.argmin(distance_matrix, axis=-1)  # [bs, n_xyz_pcld]
    corr_mesh = tf.gather(A, corr_index, batch_dims=1)  # [bs, x_xyz_pcld, 3]
    batch_B = tf.repeat(tf.expand_dims(B, axis=0), repeats=bs, axis=0)
    batch_R_svd, batch_t_svd = batch_rt_svd_transform(A=corr_mesh, B=batch_B, weights_vector=weights_vector)
    return batch_R_svd, batch_t_svd


@tf.function
def tf_icp(initial_R, initial_t, A, B, iters=3):
    """
    :param iters: iteration times for refinement
    :param initial_pose: Rt matrix [3, 4]
    :param A: [n, 3]
    :param B: can be entire point cloud from the scene or masked point cloud using predicted semantics
    :return: refined pose R, t
    """
    R = tf.cast(initial_R, tf.float32)
    t = tf.cast(initial_t, tf.float32)

    for i in range(iters):
        R_svd, t_svd = one_step_svd(R, t, A, B)
        R = tf.matmul(R_svd, R)
        t = tf.squeeze(tf.matmul(R_svd, tf.reshape(t, shape=(3, 1)))) + t_svd
    return R, t


@tf.function
def batch_tf_icp(batch_R, batch_t, A, B, k, weights_vector):
    """
    :param weights_vector:
    :param k:
    :param batch_initial_pose: a batch of Rt matrices [bs, 3, 4]
    :param iters: iteration times for refinement
    :param A: [n, 3]
    :param B: can be entire point cloud from the scene or masked point cloud using predicted semantics
    :return: refined pose R, t
    """

    bs, _, _ = batch_R.shape
    n_xyz_pcld, c = B.shape
    n_xyz_mesh, c = A.shape

    batch_R_svd, batch_t_svd = batch_one_step_svd(batch_R, batch_t, A, B, weights_vector)
    batch_R = tf.matmul(batch_R_svd, batch_R)
    batch_t = tf.reshape(tf.matmul(batch_R_svd, tf.reshape(batch_t, shape=(-1, 3, 1))) + batch_t_svd, shape=(bs, 3))

    batch_t_repeats = tf.repeat(tf.expand_dims(batch_t, axis=1), repeats=n_xyz_mesh, axis=1)
    A = tf.matmul(A, tf.transpose(batch_R, perm=(0, 2, 1))) + batch_t_repeats  # bs, N, 3
    B_repeats = tf.repeat(tf.expand_dims(B, axis=1), repeats=n_xyz_mesh, axis=1)  # [n_xyz_pcld, n_xyz_mesh, 3]
    A_repeats = tf.repeat(tf.expand_dims(A, axis=1), repeats=n_xyz_pcld, axis=1)  # [bs, n_xyz_pcld, n_xyz_mesh, 3]
    distance_matrix = tf.linalg.norm(tf.subtract(A_repeats, B_repeats), axis=-1)  # [bs, n_xyz_pcld, n_xyz_mesh]

    corr_index = tf.math.argmin(distance_matrix, axis=-1)
    distance_matrix = tf.gather(distance_matrix, corr_index, batch_dims=2)
    distance_matrix = tf.multiply(distance_matrix, weights_vector)

    distance_sum = tf.reduce_sum(distance_matrix, axis=-1)

    if bs == 1:
        min_index = tf.constant(0, dtype=tf.int64)
        R = batch_R
        t = batch_t

    else:
        min_index = tf.math.argmin(distance_sum, axis=0)
        values, indices = tf.math.top_k(-1 * distance_sum, k=k - 1)

        if min_index != 0:
            min_index = tf.constant(1, dtype=tf.int64)

        R = tf.gather(batch_R, indices)
        t = tf.gather(batch_t, indices)

        R = tf.concat([[batch_R[0]], R], axis=0)  # always including the initial_pose
        t = tf.concat([[batch_t[0]], t], axis=0)

    return R, t, min_index


@tf.function
def batch_tf_icp_2(batch_R, batch_t, A, B, weights_vector):
    """
    :param weights_vector:
    :param k:
    :param batch_initial_pose: a batch of Rt matrices [bs, 3, 4]
    :param iters: iteration times for refinement
    :param A: [n, 3]
    :param B: can be entire point cloud from the scene or masked point cloud using predicted semantics
    :return: refined pose R, t
    """

    bs, _, _ = batch_R.shape
    n_xyz_pcld, c = B.shape
    n_xyz_mesh, c = A.shape

    batch_R_svd, batch_t_svd = batch_one_step_svd(batch_R, batch_t, A, B, weights_vector)
    batch_R = tf.matmul(batch_R_svd, batch_R)
    batch_t = tf.reshape(tf.matmul(batch_R_svd, tf.reshape(batch_t, shape=(-1, 3, 1))) + batch_t_svd, shape=(bs, 3))

    batch_t_repeats = tf.repeat(tf.expand_dims(batch_t, axis=1), repeats=n_xyz_mesh, axis=1)
    A = tf.matmul(A, tf.transpose(batch_R, perm=(0, 2, 1))) + batch_t_repeats  # bs, N, 3
    B_repeats = tf.repeat(tf.expand_dims(B, axis=1), repeats=n_xyz_mesh, axis=1)  # [n_xyz_pcld, n_xyz_mesh, 3]
    A_repeats = tf.repeat(tf.expand_dims(A, axis=1), repeats=n_xyz_pcld, axis=1)  # [bs, n_xyz_pcld, n_xyz_mesh, 3]
    distance_matrix = tf.linalg.norm(tf.subtract(A_repeats, B_repeats), axis=-1)  # [bs, n_xyz_pcld, n_xyz_mesh]

    corr_index = tf.math.argmin(distance_matrix, axis=-1)
    distance_matrix = tf.gather(distance_matrix, corr_index, batch_dims=2)
    distance_matrix = tf.multiply(distance_matrix, weights_vector)

    distance_sum = tf.reduce_sum(distance_matrix, axis=-1)

    min_index = tf.math.argmin(distance_sum, axis=0)

    R_best = batch_R[min_index]
    t_best = batch_t[min_index]

    R = tf.concat([[batch_R[0]], [R_best]], axis=0)  # always including the first_one and the best one
    t = tf.concat([[batch_t[0]], [t_best]], axis=0)

    return R, t


@tf.function()
def get_Rt_varying_matrices2(R_top, t_top, A, B, weights_vector, radius=0.05, batch_size=256, factor=1.0,
                            angle_bound=0.15):
    """
    need test
    :param initial_pose:
    :param A:
    :param B:
    :param radius:
    :param batch_size:
    :return:
    """

    R = R_top[-1]
    t = t_top[-1]

    n, _, _ = R_top.shape

    n_xyz_pcld, c = B.shape
    n_xyz_mesh, c = A.shape

    A = tf.matmul(A, tf.transpose(R, perm=(1, 0))) + t  # [N, 3]
    B_repeats = tf.repeat(tf.expand_dims(B, axis=1), repeats=n_xyz_mesh, axis=1)  # [n_xyz_pcld, n_xyz_mesh, 3]
    A_repeats = tf.repeat(tf.expand_dims(A, axis=0), repeats=n_xyz_pcld, axis=0)  # [n_xyz_pcld, n_xyz_mesh, 3]
    distance_matrix = tf.linalg.norm(tf.subtract(B_repeats, A_repeats), axis=-1)  # [n_xyz_pcld, n_xyz_mesh]
    corr_index = tf.argmin(distance_matrix, axis=-1)  # [n_xyz_pcld]
    distance_corres = tf.gather(distance_matrix, corr_index, batch_dims=1)  # [n_xyz_pcld]

    num_non_zeros = tf.math.count_nonzero(weights_vector, dtype=tf.float32)

    distance_corres = tf.multiply(distance_corres, weights_vector)

    std_trans_xyz = (tf.reduce_sum(distance_corres) / num_non_zeros) * factor  # (1,)

    std_angle_xyz = tf.minimum(tf.math.asin(tf.minimum(std_trans_xyz / radius, 1.0)),
                               angle_bound)  # 0.1 bounding the rotation in -5.7 ~ 5.7 degrees

    num_random_samples = batch_size - n

    batch_xyz_angle = tf.random.uniform(shape=(3, num_random_samples, 1), minval=-1 * std_angle_xyz, maxval=std_angle_xyz)
    batch_t_translate = tf.random.uniform(shape=(num_random_samples, 3), minval=-1 * std_trans_xyz, maxval=std_trans_xyz)
    batch_cos_xyz = tf.math.cos(batch_xyz_angle)
    batch_sin_xyz = tf.math.sin(batch_xyz_angle)
    ones = tf.ones(shape=(num_random_samples, 1), dtype=tf.float32)
    zeros = tf.zeros(shape=(num_random_samples, 1), dtype=tf.float32)

    batch_R_x = tf.reshape(tf.concat([ones, zeros, zeros,
                                      zeros, batch_cos_xyz[0], -1 * batch_sin_xyz[0],
                                      zeros, batch_sin_xyz[0], batch_cos_xyz[0]], axis=1), shape=(-1, 3, 3))

    batch_R_y = tf.reshape(tf.concat([batch_cos_xyz[1], zeros, batch_sin_xyz[1],
                                      zeros, ones, zeros,
                                      -1 * batch_sin_xyz[1], zeros, batch_cos_xyz[1]], axis=1), shape=(-1, 3, 3))

    batch_R_z = tf.reshape(tf.concat([batch_cos_xyz[2], -1 * batch_sin_xyz[2], zeros,
                                      batch_sin_xyz[2], batch_cos_xyz[2], zeros,
                                      zeros, zeros, ones], axis=1), shape=(-1, 3, 3))

    batch_R_matrix = tf.matmul(batch_R_z, tf.matmul(batch_R_y, batch_R_x))

    batch_R_matrix_vary = tf.matmul(batch_R_matrix, R)
    batch_t_translate_vary = tf.squeeze(
        tf.matmul(batch_R_matrix_vary, tf.reshape(batch_t_translate, shape=(-1, 3, 1)))) + batch_t_translate

    variation_R = tf.concat([R_top, batch_R_matrix_vary], axis=0)
    variation_t = tf.concat([t_top, batch_t_translate_vary], axis=0)

    return variation_R, variation_t


@tf.function()
def get_Rt_varying_matrices(R_top, t_top, A, B, weights_vector, radius=0.05, batch_size=256, factor=1.0,
                            angle_bound=0.15, distribution='normal'):
    """
    need test
    :param initial_pose:
    :param A:
    :param B:
    :param radius:
    :param batch_size:
    :return:
    """

    R = R_top[0]
    t = t_top[0]

    n, _, _ = R_top.shape

    n_xyz_pcld, c = B.shape
    n_xyz_mesh, c = A.shape

    A = tf.matmul(A, tf.transpose(R, perm=(1, 0))) + t  # [N, 3]
    B_repeats = tf.repeat(tf.expand_dims(B, axis=1), repeats=n_xyz_mesh, axis=1)  # [n_xyz_pcld, n_xyz_mesh, 3]
    A_repeats = tf.repeat(tf.expand_dims(A, axis=0), repeats=n_xyz_pcld, axis=0)  # [n_xyz_pcld, n_xyz_mesh, 3]
    distance_matrix = tf.linalg.norm(tf.subtract(B_repeats, A_repeats), axis=-1)  # [n_xyz_pcld, n_xyz_mesh]
    corr_index = tf.argmin(distance_matrix, axis=-1)  # [n_xyz_pcld]
    distance_corres = tf.gather(distance_matrix, corr_index, batch_dims=1)  # [n_xyz_pcld]

    num_non_zeros = tf.math.count_nonzero(weights_vector, dtype=tf.float32)

    distance_corres = tf.multiply(distance_corres, weights_vector)

    std_trans_xyz = (tf.reduce_sum(distance_corres) / num_non_zeros) * factor  # (1,)

    std_angle_xyz = tf.minimum(tf.math.asin(tf.minimum(std_trans_xyz / radius, 1.0)),
                               angle_bound)  # 0.1 bounding the rotation in -5.7 ~ 5.7 degrees

    num_random_samples = batch_size - n

    batch_x_angle = tf.random.uniform(shape=(num_random_samples,), minval=-1 * std_angle_xyz, maxval=std_angle_xyz)
    batch_y_angle = tf.random.uniform(shape=(num_random_samples,), minval=-1 * std_angle_xyz, maxval=std_angle_xyz)
    batch_z_angle = tf.random.uniform(shape=(num_random_samples,), minval=-1 * std_angle_xyz, maxval=std_angle_xyz)
    batch_x_translate = tf.random.uniform(shape=(num_random_samples, 1), minval=-1 * std_trans_xyz,
                                          maxval=std_trans_xyz)
    batch_y_translate = tf.random.uniform(shape=(num_random_samples, 1), minval=-1 * std_trans_xyz,
                                          maxval=std_trans_xyz)
    batch_z_translate = tf.random.uniform(shape=(num_random_samples, 1), minval=-1 * std_trans_xyz,
                                          maxval=std_trans_xyz)

    batch_angle_xyz = tf.concat([[batch_x_angle], [batch_y_angle], [batch_z_angle]], axis=0)
    batch_sin_xyz = tf.expand_dims(tf.math.sin(batch_angle_xyz), axis=-1)  # [3, n, 3]
    batch_cos_xyz = tf.expand_dims(tf.math.cos(batch_angle_xyz), axis=-1)  # [3, n, 3]
    batch_t_translate = tf.concat([batch_x_translate, batch_y_translate, batch_z_translate], axis=1)

    ones = tf.ones(shape=(num_random_samples, 1), dtype=tf.float32)
    zeros = tf.zeros(shape=(num_random_samples, 1), dtype=tf.float32)

    batch_R_x = tf.reshape(tf.concat([ones, zeros, zeros,
                                      zeros, batch_cos_xyz[0], -1 * batch_sin_xyz[0],
                                      zeros, batch_sin_xyz[0], batch_cos_xyz[0]], axis=1), shape=(-1, 3, 3))

    batch_R_y = tf.reshape(tf.concat([batch_cos_xyz[1], zeros, batch_sin_xyz[1],
                                      zeros, ones, zeros,
                                      -1 * batch_sin_xyz[1], zeros, batch_cos_xyz[1]], axis=1), shape=(-1, 3, 3))

    batch_R_z = tf.reshape(tf.concat([batch_cos_xyz[2], -1 * batch_sin_xyz[2], zeros,
                                      batch_sin_xyz[2], batch_cos_xyz[2], zeros,
                                      zeros, zeros, ones], axis=1), shape=(-1, 3, 3))

    batch_R_matrix = tf.matmul(batch_R_z, tf.matmul(batch_R_y, batch_R_x))

    batch_R_matrix_vary = tf.matmul(batch_R_matrix, R)
    batch_t_translate_vary = tf.squeeze(
        tf.matmul(batch_R_matrix_vary, tf.reshape(batch_t_translate, shape=(-1, 3, 1)))) + batch_t_translate

    variation_R = tf.concat([batch_R_matrix_vary, R_top], axis=0)
    variation_t = tf.concat([batch_t_translate_vary, t_top], axis=0)

    return variation_R, variation_t


@tf.function()  # order
def get_Rt_varying_matrices_top(R_top, t_top, A, B, weights_vector, radius=0.05, batch_size_per_pose=32, factor=1.0,
                                angle_bound=0.15):
    """
    need test
    :param initial_pose:
    :param A:
    :param B:
    :param radius:
    :param batch_size:
    :return:
    """

    R = R_top  # k ,3, 3
    t = t_top  # k, 3

    n, _, _ = R_top.shape

    n_xyz_pcld, c = B.shape
    n_xyz_mesh, c = A.shape

    t = tf.repeat(tf.expand_dims(t, axis=1), axis=1, repeats=n_xyz_mesh)  # [k, n_mesh, 3]
    A = tf.matmul(A, tf.transpose(R, perm=(0, 2, 1))) + t  # [k, mesh, 3]

    B_repeats = tf.repeat(tf.expand_dims(B, axis=1), repeats=n_xyz_mesh, axis=1)  # [n_xyz_pcld, n_xyz_mesh, 3]
    A_repeats = tf.repeat(tf.expand_dims(A, axis=1), repeats=n_xyz_pcld, axis=1)  # [n, n_xyz_pcld, n_xyz_mesh, 3]

    distance_matrix = tf.linalg.norm(tf.subtract(B_repeats, A_repeats), axis=-1)  # [n, n_xyz_pcld, n_xyz_mesh]
    corr_index = tf.argmin(distance_matrix, axis=-1)  # [n, n_xyz_pcld]

    distance_corres = tf.gather(distance_matrix, corr_index, batch_dims=2)  # [n, n_xyz_pcld]

    num_non_zeros = tf.math.count_nonzero(weights_vector, dtype=tf.float32)
    distance_corres = tf.multiply(distance_corres, weights_vector)  # [n, n_xyz_pcld]

    std_trans_xyz = (tf.reduce_sum(distance_corres, axis=1) / num_non_zeros) * factor  # (n,)
    std_angle_xyz = tf.minimum(tf.math.asin(tf.minimum(std_trans_xyz / radius, 1.0)), angle_bound)  # (n, )

    num_random_samples = batch_size_per_pose - 1

    batch_angle_xyz = tf.random.uniform(shape=(num_random_samples, 3, 1), minval=-1 * std_angle_xyz,
                                        maxval=std_angle_xyz)  # [n_samples, 3, k]
    batch_t_translate = tf.random.uniform(shape=(num_random_samples, 3, 1), minval=-1 * std_trans_xyz,
                                          maxval=std_trans_xyz)  # [n_samples, 3, k]

    batch_angle_xyz = tf.transpose(batch_angle_xyz, perm=(1, 2, 0))  # [3, k, n_samples]
    batch_t_translate = tf.transpose(batch_t_translate, perm=(2, 0, 1))  # [k, n_samples, 3]

    batch_sin_xyz = tf.reshape(tf.math.sin(batch_angle_xyz), shape=(3, -1, 1))  # [3, n_samples*k, 1]
    batch_cos_xyz = tf.reshape(tf.math.cos(batch_angle_xyz), shape=(3, -1, 1))  # [3, n_samples*k, 1]

    ones = tf.ones(shape=(num_random_samples * n, 1), dtype=tf.float32)
    zeros = tf.zeros(shape=(num_random_samples * n, 1), dtype=tf.float32)

    batch_R_x = tf.reshape(tf.concat([ones, zeros, zeros,
                                      zeros, batch_cos_xyz[0], -1 * batch_sin_xyz[0],
                                      zeros, batch_sin_xyz[0], batch_cos_xyz[0]], axis=1),
                           shape=(-1, 3, 3))  # [n_samples*k, 3, 3]

    batch_R_y = tf.reshape(tf.concat([batch_cos_xyz[1], zeros, batch_sin_xyz[1],
                                      zeros, ones, zeros,
                                      -1 * batch_sin_xyz[1], zeros, batch_cos_xyz[1]], axis=1), shape=(-1, 3, 3))

    batch_R_z = tf.reshape(tf.concat([batch_cos_xyz[2], -1 * batch_sin_xyz[2], zeros,
                                      batch_sin_xyz[2], batch_cos_xyz[2], zeros,
                                      zeros, zeros, ones], axis=1), shape=(-1, 3, 3))

    batch_R_matrix = tf.matmul(batch_R_z, tf.matmul(batch_R_y, batch_R_x))
    batch_R_matrix = tf.reshape(batch_R_matrix, shape=(n, -1, 3, 3))  # [k, n_samples, 3, 3]

    R_top_expand = tf.repeat(tf.expand_dims(R_top, axis=1), repeats=num_random_samples, axis=1)
    batch_R_matrix_vary = tf.matmul(batch_R_matrix, R_top_expand)  # [n, n_samples, 3, 3]

    batch_t_translate_vary = tf.squeeze(
        tf.matmul(batch_R_matrix_vary,
                  tf.reshape(batch_t_translate, shape=(n, -1, 3, 1)))) + batch_t_translate  # [k, n_samples, 3]

    batch_R_matrix_vary = tf.reshape(batch_R_matrix_vary, shape=(-1, 3, 3))

    batch_t_translate_vary = tf.reshape(batch_t_translate_vary, shape=(-1, 3))

    variation_R = tf.concat([R_top, batch_R_matrix_vary], axis=0)
    variation_t = tf.concat([t_top, batch_t_translate_vary], axis=0)

    return variation_R, variation_t