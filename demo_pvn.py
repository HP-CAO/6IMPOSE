import os
import pickle
import argparse
import cv2
import numpy as np
import tqdm

from utils import *
from lib.utils import ensure_fd
from lib.net.pvn3d import Pvn3dNet
from lib.data.unity.unity_settings import UnitySettings
from lib.data.utils import get_pointxyz_unity, pcld_processor, get_label_color, get_unity_depth_value
from lib.geometry.geometry import get_pt_candidates, pts_clustering, rt_linear_fit
from lib.monitor.visualizer import project2img
from lib.geometry.geometry import pts_clustering_with_std, rt_svd_transform


def downsample_rgb(rgb, target_size=(480, 480)):
    h, w, c = rgb.shape

    assert h % target_size[0] == 0, "invalid rgb input shape"
    assert w % target_size[1] == 0, "invalid rgb input shape"

    down_factor = int(h / target_size[0])
    rgb_down = cv2.resize(rgb, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    return rgb_down, down_factor


def expand_dim(*argv):
    item_lst = []
    for item in argv:
        item = np.expand_dims(item, axis=0)
        item_lst.append(item)
    return item_lst


def demo_pvn(p, demo_path, result_save_path):
    rgb_folder = os.path.join(demo_path, 'preprocessed/rgb')
    depth_folder = os.path.join(demo_path, 'preprocessed/depth')
    result_save_folder = result_save_path
    ensure_fd(result_save_folder)

    img_lst = os.listdir(rgb_folder)

    # crop_bbox_file = os.path.join(demo_path, "crop_ofst.txt")
    # crop_bboxes = np.loadtxt(crop_bbox_file).astype(np.int)
    # crop_index = crop_bboxes[0:2]

    crop_info_folder = os.path.join(demo_path, "preprocessed/crop_info")

    # ==================== build & pvn_3d_model ====================
    data_config = UnitySettings(p.dataset_params.data_name,
                                p.dataset_params.cls_type,
                                p.dataset_params.use_data_preprocessed,
                                p.dataset_params.crop_image)

    pvn3d_model = Pvn3dNet(p.pvn3d_params,
                           rgb_input_shape=data_config.rgb_input_shape,
                           num_kpts=data_config.n_key_points,
                           num_cls=data_config.n_classes,
                           num_cpts=data_config.n_ctr_points,
                           dim_xyz=data_config.dim_pcld_xyz)

    pvn3d_model.load_weights(p.monitor_params.weights_path)

    # ==================== load_mesh_file ====================
    obj_id = data_config.obj_dict[data_config.cls_type]
    obj_color = get_label_color(obj_id, data_config.n_classes)
    unity_mesh_points = get_pointxyz_unity(obj_id)
    unity_kpts_path = os.path.join(data_config.kps_dir, "obj_{}/farthest.txt".format(obj_id))
    unity_ctr_path = os.path.join(data_config.kps_dir, "obj_{}/center.txt".format(obj_id))
    unity_key_points = np.loadtxt(unity_kpts_path)
    unity_center = np.loadtxt(unity_ctr_path)[np.newaxis, :]
    unity_mesh_kpts = np.concatenate([unity_key_points, unity_center], axis=0)
    # unity_mesh_kpts[:, 0] *= -1  # flip x-value due to the left-hand system in the unity according to the rest result do not need to flip

    for i in tqdm.tqdm(range(int(len(img_lst) * 0.8), len(img_lst))):
    # for i in tqdm.tqdm(range(int(len(img_lst) - len(img_lst) * 0.8))):
        bgr = cv2.imread(os.path.join(rgb_folder, "{:04}.png".format(i)))
        # dpt = cv2.imread(os.path.join(depth_folder, "{:04}.png".format(i)))
        # dpt = get_unity_depth_value(dpt)

        dpt_path = os.path.join(depth_folder, "{:04}.bin".format(i))
        dpt = pickle.load(open(dpt_path, "rb"))

        crop_info_path = os.path.join(crop_info_folder, '{:04}.bin'.format(i))
        crop_info = pickle.load(open(crop_info_path, "rb"))
        crop_index = crop_info[0:2]

        rgb = bgr.copy()[:, :, ::-1]

        pcld_xyz, pcld_feats, sampled_index = \
            pcld_processor(dpt, rgb, data_config.intrinsic_matrix, data_config.camera_scale,
                           p.pvn3d_params.point_net2_params.n_sample_points, crop_index)

        rgb_down, down_factor = downsample_rgb(rgb)

        input_data = expand_dim(rgb_down, down_factor, pcld_xyz, pcld_feats, sampled_index)

        kp_pre_ofst, seg_pre, cp_pre_ofst = \
            pvn3d_adp_forward_pass(input_data, pvn3d_model, training=False)

        kpts_can, ctrs_can = get_pt_candidates(pcld_xyz, kp_pre_ofst, seg_pre, cp_pre_ofst)

        # kpts_voted = pts_clustering(kpts_can, ctrs_can, bandwidth=None)
        #
        # Rt = rt_linear_fit(np.array(unity_mesh_kpts), np.array(kpts_voted))

        obj_kpts = np.concatenate([kpts_can, [ctrs_can]], axis=0)
        kpts_voted = pts_clustering_with_std(obj_kpts)
        R, t = rt_svd_transform(np.array(unity_mesh_kpts, dtype=np.float32), np.array(kpts_voted, dtype=np.float32))

        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = t

        bgr = project2img(unity_mesh_points, Rt, bgr, data_config.intrinsic_matrix,
                          data_config.camera_scale, obj_color, crop_index)
        img_save_path = os.path.join(result_save_folder, "demo_{:04}.png".format(i))
        cv2.imwrite(img_save_path, bgr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/pvn3d_480_heavy.json', help='Path to config file')
    parser.add_argument('--weights', default='./model/pvn_480_heavy_4.5k/pvn_480_heavy_4.5k_best', help='Path to pretrained weights')
    parser.add_argument('--id', default='demo', help='overrides the logfile name and the save name')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--demo_path', default='./dataset/unity/unity/img_crop/06/', help="path to demo images")
    parser.add_argument('--result_path', default='./dataset/unity/unity/img_crop/06/', help="path to demo images")

    args = parser.parse_args()

    assert args.config is not None, "config is not given"
    assert args.weights is not None, "pre_trained model is not given"

    params = read_config(args.config)
    params.monitor_params.weights_path = args.weights
    params.monitor_params.model_name = args.id
    params.monitor_params.log_file_name = args.id

    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     exit("GPU allocated failed")

    demo_pvn(params, args.demo_path, args.result_path)
