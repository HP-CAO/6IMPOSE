import os
import cv2
import argparse
import numpy as np


from utils import *
from lib.net.pvn3d import Pvn3dNet
from lib.monitor.evaluator import pvn3d_forward_pass
from lib.data.blender.blender_settings import BlenderSettings
from lib.data.utils import load_mesh, pcld_processor, \
    expand_dim, get_data_preprocessed_blender_unity
from lib.geometry.geometry import get_pt_candidates, pts_clustering_with_std, \
    rt_svd_transform

from lib.monitor.visualizer import project2img


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def eval_pvn(p, demo_path):

    save_eval_result = './eval_result/'

    # ==================== build & pvn_3d_model ====================
    data_config = BlenderSettings(p.dataset_params.data_name,
                                  p.dataset_params.cls_type,
                                  p.dataset_params.use_data_preprocessed,
                                  p.dataset_params.crop_image)

    pvn3d_model = Pvn3dNet(p.pvn3d_params,
                           rgb_input_shape=data_config.rgb_input_shape,
                           num_pts=data_config.n_sample_points,
                           num_kpts=data_config.n_key_points,
                           num_cls=data_config.n_classes,
                           num_cpts=data_config.n_ctr_points,
                           dim_xyz=data_config.dim_pcld_xyz)

    pvn3d_model.load_weights(p.monitor_params.weights_path)

    # obj_id = data_config.obj_dict[data_config.cls_type]

    mesh_path = os.path.join(data_config.mesh_dir, "{}.ply".format(data_config.cls_type))
    mesh_points = load_mesh(mesh_path, scale=0.01) * data_config.mesh_factor
    kpts_path = os.path.join(data_config.kps_dir, "{}/farthest.txt".format(data_config.cls_type))
    corner_path = os.path.join(data_config.kps_dir, "{}/corners.txt".format(data_config.cls_type))
    key_points = np.loadtxt(kpts_path)
    center = [np.loadtxt(corner_path).mean(0)]
    mesh_kpts = np.concatenate([key_points, center], axis=0) * data_config.mesh_factor
    random_image = np.random.randint(low=p.dataset_params.train_size, high=p.dataset_params.size_all, size=1)
    preprocessed_data_path = os.path.join(data_config.preprocessed_folder, "{:05}.bin".format(random_image))
    RT_gt, rgb, dpt, cam_intrinsic, crop_index = get_data_preprocessed_blender_unity(preprocessed_data_path)
    xy_offset = crop_index[:2]
    pcld_xyz, pcld_feats, sampled_index = \
        pcld_processor(dpt, rgb, cam_intrinsic, data_config.camera_scale,
                       data_config.n_sample_points, xy_offset)
    input_data = expand_dim(rgb, pcld_xyz, pcld_feats, sampled_index)

    kp_pre_ofst, seg_pre, cp_pre_ofst = \
        pvn3d_forward_pass(input_data, pvn3d_model, training=False)

    kpts_can, ctrs_can = get_pt_candidates(pcld_xyz, kp_pre_ofst, seg_pre, cp_pre_ofst)

    obj_kpts = np.concatenate([kpts_can, [ctrs_can]], axis=0)
    kpts_voted = pts_clustering_with_std(obj_kpts)
    R, t = rt_svd_transform(np.array(mesh_kpts, dtype=np.float32), np.array(kpts_voted, dtype=np.float32))

    Rt_pre = np.zeros((3, 4))
    Rt_pre[:, :3] = R
    Rt_pre[:, 3] = t
    rgb = project2img(mesh_points, Rt_pre, rgb, cam_intrinsic,
                      data_config.camera_scale, [0, 0, 255], xy_offset)

    img_save_path = os.path.join(save_eval_result, "eval_{:05}.png".format(random_image))
    cv2.imwrite(img_save_path, rgb[:, :, ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/pvn3d_blender_evaluation.json', help='Path to config file')
    parser.add_argument('--weights', default='./models/pvn3d_blender_best/pvn',
                        help='Path to pretrained weights')
    parser.add_argument('--id', default='demo', help='overrides the logfile name and the save name')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--demo_path', default='./dataset/linemod/linemod/data/09', help="path to demo images")

    args = parser.parse_args()

    assert args.config is not None, "config is not given"
    assert args.weights is not None, "pre_trained model is not given"

    params = read_config(args.config)
    params.monitor_params.weights_path = args.weights
    params.monitor_params.model_name = args.id
    params.monitor_params.log_file_name = args.id

    eval_pvn(params, args.demo_path)
