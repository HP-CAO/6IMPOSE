import os
import cv2
import sys
import inspect
import io

import matplotlib.pyplot as plt
from scipy import interpolate

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import argparse
import numpy as np
import tqdm
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import AveragePooling2D, MaxPool2D
from utils import *
from lib.net.pvn3d import Pvn3dNet
from lib.data.blender.blender_settings import BlenderSettings
from lib.net.utils import match_choose
from lib.data.utils import load_mesh, pcld_processor, \
    expand_dim

import pickle

"""
Here we set up the testing pipeline for blender dataset
"""


def get_data_preprocessed_gt(preprocessed_data_path):
    preprocessed_data = pickle.load(open(preprocessed_data_path, "rb"))
    label_info = preprocessed_data['label_info']
    kpts_targ_offst = preprocessed_data['kpts_targ_offst']
    ctr_targ_offst = preprocessed_data['ctr_targ_offst']

    return label_info, kpts_targ_offst, ctr_targ_offst


def get_data_preprocessed(preprocessed_data_path):
    preprocessed_data = pickle.load(open(preprocessed_data_path, "rb"))
    rgb = preprocessed_data['rgb']
    crop_info = preprocessed_data['crop_info']
    rt = preprocessed_data['RT']
    if 'depth' in preprocessed_data.keys():
        depth = preprocessed_data['depth']
    else:
        depth = None
    cam_intrinsic = preprocessed_data['K']
    crop_index, crop_down_factor = crop_info
    pcld_info = preprocessed_data['pcld_info']
    pcld_xyz_rgb_nm, sampled_index = pcld_info

    return rt, rgb, depth, cam_intrinsic, crop_index, pcld_xyz_rgb_nm[:, :3], pcld_xyz_rgb_nm[:, 3:], sampled_index


class TestParams:
    def __init__(self):
        self.use_icp = True
        self.save_results = True


def pvn3d_forward_pass_feature(input_data, pvn3d_feature_model, training=False):
    """
    params:
            rgb [bs, H, W, 3]
            pcld_xyz [bs, npts, 3]
            pcld_feats [bs, npts, 6]
            sampled_index [bs, npts]
    output:
           kpts_pre [bs, npts, n_kpts, 3]
           cpts_pre [bs, npts, n_cpts, 3]
           segs_pre [bs, npts, n_cls]
    """

    rgb, pcld_xyz, pcld_feats, sampled_index = input_data

    pcld = [pcld_xyz, pcld_feats]
    inputs = [pcld, sampled_index, rgb]
    rgb_features, pcld_emb, rgb_emb = pvn3d_feature_model(inputs, training=training)

    return rgb_features, pcld_emb, rgb_emb


def visualize_features_map(feature_array):
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
    return

    # show the figure
    # fig.savefig('features.png')  # todo change the resolution


def get_cluster_index(mesh_points, cluster_centers, seg_index, cluster_dis=0.015):
    cluster_index_dic = {}
    for i, center in enumerate(cluster_centers):
        dis_norm = np.linalg.norm(mesh_points - np.array([center]), axis=-1)
        cluster_index = np.where(dis_norm < cluster_dis)[0]
        cluster_index_dic['cluster_{}'.format(i)] = seg_index[cluster_index]
    return cluster_index_dic


def rescale_map(map):
    map = (map - np.nanmean(map)) / (np.nanmax(map) - np.nanmin(map))
    return map


def mapping_pcld_2_mesh(seg_index, pcld_xyz, RT_gt):
    obj_pcld = pcld_xyz[seg_index]
    mesh_points = np.dot((obj_pcld - RT_gt[:, 3]), np.linalg.inv(RT_gt[:, :3].T))
    return mesh_points


def eval_pvn(p, test_params: TestParams):
    save_eval_result = './eval_result/'

    # ==================== build & pvn_3d_model ====================
    data_config = BlenderSettings(p.dataset_params.data_name,
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

    # obj_id = data_config.obj_dict[data_config.cls_type]

    # mesh_path = os.path.join(data_config.mesh_dir, "{}.ply".format(data_config.cls_type))
    # mesh_points = load_mesh(mesh_path, scale=0.01) * data_config.mesh_factor
    kpts_path = os.path.join(data_config.kps_dir, "{}/farthest.txt".format(data_config.cls_type))
    key_points = np.loadtxt(kpts_path) * data_config.mesh_factor
    image_id = np.random.randint(p.dataset_params.train_size, p.dataset_params.size_all - 1000)

    # preprocessed_data_path = os.path.join(data_config.preprocessed_folder, "{:06}.bin".format(image_id))
    preprocessed_data_path = os.path.join('dataset/linemod/linemod/data/09/preprocessed/00001.bin')
    label_info, _, _ = get_data_preprocessed_gt(preprocessed_data_path)
    label_list, _ = label_info

    RT_gt, rgb, dpt, cam_intrinsic, crop_index, pcld_xyz, pcld_feats, sampled_index \
        = get_data_preprocessed(preprocessed_data_path)

    segs_mask = np.argmax(label_list, axis=-1).squeeze().astype(np.bool)
    seg_index = np.where(segs_mask > 0)[0]

    # xy_offset = crop_index[:2]
    num_to_vis = 16
    input_data = expand_dim(rgb, pcld_xyz, pcld_feats, sampled_index)
    resnet_features = pvn3d_model.resnet_model(input_data[0])
    tf_rgb_features = pvn3d_model.psp_model(resnet_features)
    rgb_features = tf_rgb_features.numpy().squeeze()
    # visualize_features_map(rgb_features[:, :, :num_to_vis])

    tf_pointnet_features = pvn3d_model.pointnet2_model([input_data[1], input_data[2]])  # bs, n_pts, c
    h, w, c_rgb = rgb_features.shape
    pcld_maps = np.ones(shape=(h * w, c_rgb)) * 0.0
    pointnet_features = tf_pointnet_features.numpy().squeeze()
    # pointnet_features = rescale_map(pointnet_features)
    pcld_maps[sampled_index] = pointnet_features
    pcld_maps = np.reshape(pcld_maps, newshape=(1, h, w, -1))  # h, w, c
    # pcld_maps_ave_pooling = AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')(pcld_maps).numpy().squeeze()
    pcld_maps_ave_pooling = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(pcld_maps).numpy().squeeze()

    # visualize_features_map(pcld_maps_ave_pooling[:, :, :num_to_vis])

    rgb_emb = match_choose(tf_rgb_features, [sampled_index])
    tf_feats_fused = pvn3d_model.dense_fusion_model([rgb_emb, tf_pointnet_features], training=False)
    # only visualize first c_rgb maps, since there are too many
    feats_fused = tf_feats_fused.numpy().squeeze()
    # feats_fused = rescale_map(feats_fused)
    fused_maps = np.ones(shape=(h * w, c_rgb)) * 0.0
    fused_maps[sampled_index] = feats_fused[:, :c_rgb]
    fused_maps = np.reshape(fused_maps, newshape=(1, h, w, -1))  # h, w, c

    # fused_maps_ave_pooling = AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')(fused_maps).numpy().squeeze()
    fused_maps_ave_pooling = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(fused_maps).numpy().squeeze()

    feats_list = [rgb_features[:, :, :num_to_vis], pcld_maps_ave_pooling[:, :, :num_to_vis],
                  fused_maps_ave_pooling[:, :, :num_to_vis]]

    feats_array = np.concatenate(feats_list, axis=-1)
    visualize_features_map(feats_array)

    # mapped_mesh_points = mapping_pcld_2_mesh(seg_index, pcld_xyz, RT_gt)
    # cluster_index_dic = get_cluster_index(mapped_mesh_points, key_points, seg_index, cluster_dis=0.015)
    #     for cluster_id, cluster_indices in enumerate(cluster_index_dic.values()):
    #
    #         if len(cluster_indices) == 0:
    #             channel_mean = np.empty(256)
    #             channel_std = np.empty(256)
    #             channel_mean[:] = np.nan
    #             channel_std[:] = np.nan
    #         else:
    #             point_wise_features = np.concatenate((pcld_emb.numpy()[0], rgb_emb.numpy()[0]), axis=-1)
    #             cluster_features = point_wise_features[cluster_indices]
    #             channel_mean = cluster_features.mean(axis=0)  # n_channels
    #             channel_std = cluster_features.std(axis=0)  # n_channels
    #
    #         mean_array[i][cluster_id] = channel_mean
    #         std_array[i][cluster_id] = channel_std
    #
    # mean_cluster_wise = np.nanmean(mean_array, axis=0)  # [n_cluster, n_channels]
    # std_cluster_wise = np.nanstd(std_array, axis=0)  # [n_cluster, n_channels]
    #
    # mean_dataset_wise = np.mean(mean_cluster_wise, axis=0)  # n_channels
    # std_dataset_wise = np.std(std_cluster_wise, axis=0)  # n_channels
    #
    # ix = 0
    # row = 3
    # col = 3
    # for _ in range(row):
    #     for _ in range(col):
    #         # specify subplot and turn of axis
    #         ax = pyplot.subplot(row, col, ix + 1)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         if ix == 0:
    #             ax.plot(range(256), mean_dataset_wise)
    #             ax.plot(range(256), std_dataset_wise)
    #         else:
    #             ax.plot(range(256), mean_cluster_wise[ix - 1])
    #             ax.plot(range(256), std_cluster_wise[ix - 1])
    #         ix += 1
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/pvn3d_blender_debug.json', help='Path to config file')
    parser.add_argument('--weights', default='./models/pvn3d_blender_near_290K_best/pvn3d',
                        help='Path to pretrained weights')
    parser.add_argument('--id', default='demo', help='overrides the logfile name and the save name')
    parser.add_argument('--params', nargs='*', default=None)

    args = parser.parse_args()

    assert args.config is not None, "config is not given"
    assert args.weights is not None, "pre_trained model is not given"

    physical_devices = tf.config.list_physical_devices('GPU')

    use_gpu = True

    # if not use_gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # else:
    #     try:
    #         print(physical_devices[-1])
    #         tf.config.experimental.set_memory_growth(physical_devices[-1], True)
    #     except:
    #         exit("GPU allocated failed")

    params = read_config(args.config)
    params.monitor_params.weights_path = args.weights
    params.monitor_params.model_name = args.id
    params.monitor_params.log_file_name = args.id

    test_params = TestParams()
    eval_pvn(params, test_params)
