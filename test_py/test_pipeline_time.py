import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import time
import tqdm
import argparse
import numpy as np
import tensorflow as tf
import inspect
import sys
import cv2
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.run_functions_eagerly(True)
tf.config.experimental.set_memory_growth(gpus[0], True)

from lib.net.pvn3d_adp import Pvn3dNet
from lib.monitor.evaluator import pvn3d_forward_pass
# from pvn3d.data.blender.blender_settings import BlenderSettings
from lib.data.linemod.linemod_settings import LineModSettings
from lib.data.utils import pcld_processor_tf, expand_dim, load_mesh
from PIL import Image
from lib.geometry.geometry import get_pt_candidates, pts_clustering_with_std, \
    rt_svd_transform, icp_refinement, tf_icp, batch_tf_icp, get_Rt_varying_matrices
from lib.monitor.visualizer import project2img, vis_stochastic_poses
from lib.data.utils import get_data_preprocessed, get_mesh_diameter
from lib.net.pprocessnet import InitialPoseModel, SCPModel
from utils import *
from lib.monitor.evaluator import cal_auc, cal_accuracy, cal_add_dis, cal_adds_dis

n_sample_points = 1024

camera_matrix = np.array([
    [572.411376953125, 0.0, 320.0],
    [0.0, 573.5703735351562, 240.0],
    [0.0, 0.0, 1.0]]).astype(np.float32)

camera_scale = 1
rgb_path = "camera/test_image/rgb_1000.png"
depth_path = "camera/test_image/depth_1000.exr"
save_directory = "camera/test_result"

warming_times = 10
testing_times = 300

intrinsic_matrix = np.array([[572.4114, 0., 325.2611],
                             [0., 573.57043, 242.04899],
                             [0., 0., 1.]])


def test_stochastic_icp(mesh_points, Rt_pre, rgb, initial_R, initial_t, A, B, radius=0.05, batch_size=64, factor=0.5,
                        iters=5, k=10):
    R = tf.cast([initial_R], tf.float32)
    t = tf.cast([initial_t], tf.float32)

    for i in range(iters):
        batch_R, batch_t = get_Rt_varying_matrices(R, t, A, B, radius, batch_size, factor)
        print(batch_R.shape)
        image_list = []
        for j in range(batch_size):
            img = project2img(mesh_points, Rt_pre, rgb, intrinsic_matrix, camera_scale, (0, 255, 0), [0, 0])
            image_list.append(img)
        vis_stochastic_poses(image_list)
        save_name = os.path.join(save_directory, f"{i}_iter.png")
        plt.savefig(save_name, dpi=300)
        # cv2.imwrite(filename=save_name, img=np.array(figure))
        R, t = batch_tf_icp(batch_R, batch_t, A, B, k)


def decoding_dpt_exr(path_exr, rescale_factor=1):
    dpt = cv2.imread(path_exr, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    dpt = dpt[:, :, 2] * rescale_factor
    dpt_mask = dpt < 5  # in meters, we filter out the background( > 5m)
    dpt = dpt * dpt_mask
    return dpt


def preprocess_data():
    with Image.open(rgb_path) as rgb:
        rgb = np.array(rgb)[:, :, :3].astype(np.uint8)

    # with Image.open(depth_path) as dpt:
    #     depth = np.array(dpt)

    depth = decoding_dpt_exr(depth_path)

    rgb_normalized = rgb / 255.
    pcld_xyz, pcld_feats, sampled_id = pcld_processor_tf(depth, rgb_normalized.astype(np.float32), camera_matrix,
                                                         camera_scale, n_sample_points, xy_ofst=tf.constant((0, 0)),
                                                         depth_trunc=tf.constant(2.0))
    input_data = expand_dim(rgb, pcld_xyz, pcld_feats, sampled_id)

    return input_data, pcld_xyz


def eval_pvn_time(p, save_path):
    ensure_fd(save_path)
    run_time_mean_list = []

    # ==================== build & pvn_3d_model ====================
    data_config = LineModSettings(p.dataset_params.data_name,
                                  p.dataset_params.cls_type,
                                  p.dataset_params.use_data_preprocessed,
                                  p.dataset_params.crop_image)

    print_model_summary = True
    pvn3d_model = Pvn3dNet(p.pvn3d_params,
                           rgb_input_shape=data_config.rgb_input_shape,
                           num_kpts=data_config.n_key_points,
                           num_cls=data_config.n_classes,
                           num_cpts=data_config.n_ctr_points,
                           dim_xyz=data_config.dim_pcld_xyz)

    if p.monitor_params.weights_path is not None:
        pvn3d_model.load_weights(p.monitor_params.weights_path)

    rescale_factor = 0.001
    obj_id = data_config.obj_dict[data_config.cls_type]
    mesh_path = os.path.join(data_config.mesh_dir, "obj_{:02}.ply".format(obj_id))
    mesh_points = load_mesh(mesh_path, scale=rescale_factor, n_points=500)
    mesh_info_path = os.path.join(data_config.mesh_dir, "model_info.yml")
    mesh_diameter = get_mesh_diameter(mesh_info_path, obj_id) * rescale_factor  # from mm to m
    kpts_path = os.path.join(data_config.kps_dir, "{}/farthest.txt".format(data_config.cls_type))
    corner_path = os.path.join(data_config.kps_dir, "{}/corners.txt".format(data_config.cls_type))
    key_points = np.loadtxt(kpts_path)
    center = [np.loadtxt(corner_path).mean(0)]
    mesh_kpts = np.concatenate([key_points, center], axis=0)

    RT_gt, rgb, dpt, cam_intrinsic, crop_index, pcld_xyz, pcld_feats, sampled_index, crop_factor = \
        get_data_preprocessed('./camera/test_image/test_crop', 1000)

    n_samples = pvn3d_model.num_pts
    initial_pose_model = InitialPoseModel()

    scp_model = SCPModel(scp_radius=mesh_diameter * 0.5, scp_factor=0.5, scp_iters=1, angle_bound=1.0)
    dcp_model = SCPModel(scp_radius=mesh_diameter * 0.5, scp_batch_size=1, scp_iters=1, angle_bound=1.0)
    icp_threshold = 2

    sample_inds = tf.random.shuffle(tf.range(12288))[:n_samples]
    pcld_xyz = tf.gather(pcld_xyz, sample_inds)
    pcld_feats = tf.gather(pcld_feats, sample_inds)
    sampled_index = tf.gather(sampled_index, sample_inds)

    input_data = expand_dim(rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor)

    refine_test(mesh_points)

    # warming up:
    # for _ in tqdm.tqdm(range(warming_times)):
    #     # input_data, _ = preprocess_data()
    #     _, _, _ = pvn3d_forward_pass(input_data, pvn3d_model, training=False)
    #     if print_model_summary:
    #         print(pvn3d_model.summary())
    #         print_model_summary = False

    for _ in tqdm.tqdm(range(testing_times)):
        t0 = time.perf_counter()
        # input_data, pcld_xyz = preprocess_data()
        kp_pre_ofst, seg_pre, cp_pre_ofst = pvn3d_forward_pass(input_data, pvn3d_model, training=False)
        R_inital, t_intial, kpts_voted = initial_pose_model(
            [input_data[1], kp_pre_ofst, cp_pre_ofst, seg_pre, tf.expand_dims(mesh_kpts, axis=0)], training=False)
        Rt_pre_initial = tf.concat([R_inital[0], tf.expand_dims(t_intial[0], axis=-1)], axis=-1).numpy()

        add_score_ini = cal_add_dis(mesh_points, Rt_pre_initial, RT_gt)

        Rt_pre_refine = np.zeros((3, 4))
        use_icp = True
        num_pts_seg = tf.math.count_nonzero(seg_pre[0])
        print(num_pts_seg > icp_threshold)

        if use_icp and num_pts_seg > icp_threshold:
            '''Dcp and scp'''

            R_scp, t_scp = dcp_model([pcld_xyz, seg_pre[0], R_inital, t_intial, mesh_points])
            Rt_pre_refine[:, :3] = R_scp
            Rt_pre_refine[:, 3] = t_scp
            add_score_refine_dcp = cal_add_dis(mesh_points, Rt_pre_refine, RT_gt)

            R_scp, t_scp = scp_model([pcld_xyz, seg_pre[0], R_inital, t_intial, mesh_points])
            Rt_pre_refine[:, :3] = R_scp
            Rt_pre_refine[:, 3] = t_scp

            add_score_refine_scp = cal_add_dis(mesh_points, Rt_pre_refine, RT_gt)

            '''Open3D icp'''
            # Rt_pre = icp_refinement(initial_pose=Rt_pre, mesh_pts=mesh_points, pcld_xyz=selected_pcld)

            print(
                f"initial_pose:{add_score_ini} \n refine_pose_scp{add_score_refine_scp} \n refine_pose_dcp{add_score_refine_dcp}")

        visulize = True

        if visulize:
            img = project2img(mesh_points, Rt_pre_initial, input_data[0][0], intrinsic_matrix, camera_scale,
                              (0, 255, 0),
                              crop_index[:2])
            plt.imshow(img)
            plt.show()
            return

        delta_t = time.perf_counter() - t0
        run_time_mean_list.append(delta_t)

    mean = np.array(run_time_mean_list).mean()
    std = np.array(run_time_mean_list).std()

    print("num_samples: {}, Mean for 1000 images is {} ".format(n_samples, mean))
    print("num_samples: {}, Std for 1000 images is {} ".format(n_samples, std))


def refine_test(mesh_points):
    dcp_model = SCPModel(scp_radius=0.1 * 0.5, scp_batch_size=256, scp_iters=3, angle_bound=0.3, scp_factor=1.5)
    RT_gt = np.diag([1., 1., 1., 1.])
    initial_rt = np.copy(RT_gt)
    initial_rt[0, 3] += 0.05
    Rt_pre_refine = np.zeros((3, 4))

    R_inital = RT_gt[:3, :3]
    t_intial = RT_gt[:3, 3]
    num_points = 64

    fake_pcld = mesh_points[:num_points]
    fake_mesh = mesh_points[:num_points] + initial_rt[:3, 3]

    zeros = np.zeros(shape=(num_points, 1))
    ones = np.ones(shape=(num_points, 1))
    ones = np.concatenate([zeros, ones], axis=-1)

    R_scp, t_scp = dcp_model([fake_mesh, ones, R_inital, t_intial, fake_pcld])

    print("R_scp", R_scp)
    print("T_scp", t_scp)
    Rt_pre_refine[:, :3] = R_scp
    Rt_pre_refine[:, 3] = t_scp

    add_score = cal_add_dis(fake_mesh, Rt_pre_refine, initial_rt[:3])
    print("add_score:", add_score)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/test_config.json', help='Path to config file')
    parser.add_argument('--id', default='demo', help='overrides the logfile name and the save name')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--save_path', default='test_plot', help="path to demo images")
    parser.add_argument('--weights', default='models/test_model/pvn3d', help='Path to pretrained weights')

    args = parser.parse_args()

    assert args.config is not None, "config is not given"

    params = read_config(args.config)
    params.monitor_params.weights_path = args.weights
    params.monitor_params.model_name = args.id
    params.monitor_params.log_file_name = args.id
    save_path = os.path.join(args.save_path, args.id)

    eval_pvn_time(params, save_path)
