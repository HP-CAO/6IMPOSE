import os
import cv2
import argparse
import numpy as np
import tqdm

from utils import *
from lib.net.pvn3d import Pvn3dNet
from lib.monitor.evaluator import pvn3d_forward_pass
from lib.data.blender.blender_settings import BlenderSettings
from lib.data.utils import load_mesh, pcld_processor, \
    expand_dim, get_data_preprocessed_blender_unity
from lib.geometry.geometry import get_pt_candidates, pts_clustering_with_std, \
    rt_svd_transform, icp_refinement
from lib.monitor.evaluator import cal_auc, cal_add_dis, cal_adds_dis
from lib.monitor.visualizer import project2img
import tensorflow as tf

"""
Here we set up the testing pipeline for blender dataset
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestParams:
    def __init__(self):
        self.use_icp = True
        self.save_results = True


def eval_pvn(p, test_params: TestParams):
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
    image_ids = range(p.dataset_params.train_size, p.dataset_params.size_all-1000)
    add_score_list = []
    adds_score_list = []

    for i in tqdm.tqdm(image_ids):

        preprocessed_data_path = os.path.join(data_config.preprocessed_folder, "{:06}.bin".format(i))

        RT_gt, rgb, dpt, cam_intrinsic, crop_index, _, _, _ = get_data_preprocessed_blender(preprocessed_data_path)

        xy_offset = crop_index[:2]

        pcld_xyz, pcld_feats, sampled_index = \
            pcld_processor(dpt, rgb, cam_intrinsic, data_config.camera_scale,
                           data_config.n_sample_points, xy_offset)

        input_data = expand_dim(rgb, pcld_xyz, pcld_feats, sampled_index)

        kp_pre_ofst, seg_pre, cp_pre_ofst = \
            pvn3d_forward_pass(input_data, pvn3d_model, training=False)

        kpts_cpts_can = get_pt_candidates(pcld_xyz, kp_pre_ofst, seg_pre, cp_pre_ofst)

        kpts_voted = pts_clustering_with_std(kpts_cpts_can)

        R, t = rt_svd_transform(np.array(mesh_kpts, dtype=np.float32), np.array(kpts_voted, dtype=np.float32))

        Rt_pre = np.zeros((3, 4))
        Rt_pre[:, :3] = R
        Rt_pre[:, 3] = t

        segs = np.argmax(seg_pre.numpy(), axis=-1).squeeze()  # kill the batch dim
        obj_pts_index = np.where(segs == 1)[0]
        len_index = obj_pts_index.shape[0]

        if len_index == 0:
            continue

        if test_params.use_icp:
            Rt_pre = icp_refinement(Rt_pre, mesh_points, pcld_xyz)

        rgb = project2img(mesh_points, Rt_pre, rgb, cam_intrinsic,
                          data_config.camera_scale, [0, 0, 255], xy_offset)

        add_score = cal_add_dis(mesh_points, Rt_pre, RT_gt)
        add_score_list.append(add_score)
        adds_score = cal_adds_dis(mesh_points, Rt_pre, RT_gt)
        adds_score_list.append(adds_score)

        if test_params.save_results:
            img_save_path = os.path.join(save_eval_result, "eval_{:06}.png".format(i))
            cv2.imwrite(img_save_path, rgb[:, :, ::-1])

    add_auc = cal_auc(add_score_list, max_dis=0.1)
    adds_auc = cal_auc(adds_score_list, max_dis=0.1)
    print(add_auc, adds_auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/pvn3d_blender_evaluation.json', help='Path to config file')
    parser.add_argument('--weights', default='./models/pvn3d_blender_near_290K_best/pvn3d',
                        help='Path to pretrained weights')
    parser.add_argument('--id', default='demo', help='overrides the logfile name and the save name')
    parser.add_argument('--params', nargs='*', default=None)

    args = parser.parse_args()

    assert args.config is not None, "config is not given"
    assert args.weights is not None, "pre_trained model is not given"

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[-1], True)
    except:
        exit("GPU allocated failed")

    params = read_config(args.config)
    params.monitor_params.weights_path = args.weights
    params.monitor_params.model_name = args.id
    params.monitor_params.log_file_name = args.id

    test_params = TestParams()
    eval_pvn(params, test_params)
