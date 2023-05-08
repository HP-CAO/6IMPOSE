# This demo is the script for pvn working on the cropped image

import os

import cv2
import inspect
import sys
import argparse
from PIL import Image

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import *


def performance_eval_linemod(p):
    import numpy as np
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')

    try:
        tf.config.experimental.set_memory_growth(physical_devices[-1], True)
    except:
        exit("GPU allocated failed")

    from lib.net.pvn3d_adp import Pvn3dNet, forward_pass
    from lib.net.pprocessnet import InitialPoseModel
    from lib.data.utils import load_mesh, expand_dim, get_crop_index, crop_image, pcld_processor_tf
    from lib.monitor.visualizer import project_p3d, draw_p2ds
    from lib.data.linemod.linemod_settings import LineModSettings

    data_config = LineModSettings(p.dataset_params.data_name,
                                  p.dataset_params.cls_type,
                                  p.dataset_params.use_preprocessed,
                                  p.dataset_params.crop_image)

    resnet_w_h = 80
    resnet_input_size = [resnet_w_h, resnet_w_h]
    bbox_default = [320., 120., 400., 200.]  # hacking 80x80, this part will be done using yolo in the robot
    # experiements

    rescale_factor = 0.001
    mesh_path = "./demo_data/obj_06.ply"
    rgb_path = "./demo_data/rgb_0001.png"
    dpt_path = "./demo_data/dpt_0001.png"
    kpts_path = "./demo_data/farthest.txt"
    corner_path = "./demo_data/corners.txt"

    mesh_points = load_mesh(mesh_path, scale=rescale_factor, n_points=500)

    key_points = np.loadtxt(kpts_path)
    center = [np.loadtxt(corner_path).mean(0)]
    mesh_kpts = np.concatenate([key_points, center], axis=0)
    mesh_kpts = tf.cast(tf.expand_dims(mesh_kpts, axis=0), dtype=tf.float32)
    intrinsic_matrix = data_config.intrinsic_matrix
    n_sample_points = p.pvn3d_params.point_net2_params.n_sample_points

    """=========== Initialize deep model==========="""

    pvn3d_model = Pvn3dNet(p.pvn3d_params,
                           rgb_input_shape=[80, 80, 3],
                           num_kpts=data_config.n_key_points,
                           num_cls=data_config.n_classes,
                           num_cpts=data_config.n_ctr_points,
                           dim_xyz=data_config.dim_pcld_xyz)

    initial_pose_model = InitialPoseModel()

    if p.monitor_params.weights_path is not None:
        pvn3d_model.load_weights(p.monitor_params.weights_path)

    """=========== Input preparation ==========="""

    with Image.open(rgb_path) as rgb:
        rgb = np.array(rgb).astype(np.uint8)

    with Image.open(dpt_path) as depth:
        dpt = np.array(depth) / 1000.

    crop_index, crop_factor = get_crop_index(bbox_default, base_crop_resolution=resnet_input_size)
    rgb = crop_image(rgb, crop_index)
    depth = crop_image(dpt, crop_index)
    rgb_normalized = rgb.copy() / 255.
    pcld_xyz, pcld_feats, sampled_index = pcld_processor_tf(depth.astype(np.float32),
                                                            rgb_normalized.astype(np.float32), intrinsic_matrix, 1,
                                                            n_sample_points, xy_ofst=crop_index[:2],
                                                            depth_trunc=2.0)

    rgb = tf.image.resize(rgb, resnet_input_size).numpy()
    input_data = expand_dim(rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor)

    """=========== Deep model inference ==========="""

    kp_pre_ofst, seg_pre, cp_pre_ofst = forward_pass(input_data, pvn3d_model, training=False)
    R, t, kpts_voted = initial_pose_model([input_data[1], kp_pre_ofst, cp_pre_ofst, seg_pre, mesh_kpts], training=False)
    Rt_pre = tf.concat([R[0], tf.expand_dims(t[0], -1)], axis=-1).numpy()

    """=========== Result visualization ==========="""
    pred_pts = np.dot(mesh_points.copy(), Rt_pre[:, :3].T) + Rt_pre[:, 3]
    pre_mesh_projected = project_p3d(pred_pts, cam_scale=1, K=intrinsic_matrix)
    pre_mesh_projected[:, 0] -= crop_index[0]
    pre_mesh_projected[:, 1] -= crop_index[1]
    img_pre_projected = draw_p2ds(rgb, pre_mesh_projected, r=1, color=[0, 0, 255])
    proj_pose_path = os.path.join("result.jpg")
    cv2.imwrite(proj_pose_path, cv2.cvtColor(img_pre_projected, cv2.COLOR_RGB2BGR))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./demo_data/model/log/cat/config.json', help='Path to config file')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--weights', default='./demo_data/model/model/cat/best_model/model', help='Path to pretrained weights')
    parser.add_argument('--gpu_id', default="0")

    args = parser.parse_args()

    assert args.config is not None, "config is not given"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    params = read_config(args.config)
    params.monitor_params.weights_path = args.weights

    performance_eval_linemod(params)
