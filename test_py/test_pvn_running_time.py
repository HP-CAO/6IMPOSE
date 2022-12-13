import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time

import tqdm
import argparse
import numpy as np
import tensorflow as tf
import inspect
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.run_functions_eagerly(True)
tf.config.experimental.set_memory_growth(gpus[0], True)

from lib.net.pvn3d import Pvn3dNet
from lib.monitor.evaluator import pvn3d_forward_pass
from lib.data.blender.blender_settings import BlenderSettings
from lib.data.utils import pcld_processor, expand_dim, get_data_preprocessed_gt
from matplotlib import pyplot as plt
from lib.net.resnet import ResNetParams, ResNet
from lib.net.pspnet import PspNetParams, PspNet
# from pvn3d.net.pointnet2 import PointNet2Params, PointNetModel
from lib.net.pointnet2_tf import PointNet2Params, PointNet2TfModel


from lib.net.densefusion import DenseFusionNet, DenseFusionNetParams
from lib.net.mlp import MlpNets, MlpNetsParams
from lib.net.utils import match_choose
from utils import *
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_data_preprocessed(preprocessed_data_path):
    preprocessed_data = pickle.load(open(preprocessed_data_path, "rb"))
    rgb = preprocessed_data['rgb'] / 255.
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


def eval_pvn_time(p, save_path):
    ensure_fd(save_path)
    # ==================== build & pvn_3d_model ====================
    data_config = BlenderSettings(p.dataset_params.data_name,
                                  p.dataset_params.cls_type,
                                  p.dataset_params.use_data_preprocessed,
                                  p.dataset_params.crop_image)

    rgb_shape_list = [(480, 640, 3)]

    result_mean_list = []
    result_std_list = []

    print_model_summary = True

    for rgb_shape in rgb_shape_list:

        data_config.rgb_input_shape = rgb_shape

        # data_config.n_sample_points = 5000

        pvn3d_model = Pvn3dNet(p.pvn3d_params,
                               rgb_input_shape=data_config.rgb_input_shape,
                               num_kpts=data_config.n_key_points,
                               num_cls=data_config.n_classes,
                               num_cpts=data_config.n_ctr_points,
                               dim_xyz=data_config.dim_pcld_xyz)

        pvn3d_model.pointnet2_model.load_weights('./models/pt2_training_tf_seg_from_logits_conv_argmax_best/pt2')

        # resnet_net = ResNet(p.pvn3d_params.resnet_params, rgb_shape)
        # resnet_model = resnet_net.build_resnet_model()
        # psp_net = PspNet(p.pvn3d_params.psp_params)
        # psp_model = psp_net.build_psp_model(list(resnet_model.output_shape)[1:])

        pointnet2_model = PointNet2TfModel(p.pvn3d_params.point_net2_params, num_classes=2)

        # dense_fusion_net = DenseFusionNet(p.pvn3d_params.dense_fusion_params)

        # dense_fusion_model = dense_fusion_net.build_dense_fusion_model(
        #     rgb_emb_shape=(data_config.n_sample_points, p.pvn3d_params.dense_fusion_params.num_embeddings),
        #     pcl_emb_shape=(data_config.n_sample_points, p.pvn3d_params.dense_fusion_params.num_embeddings))

        # mlp_net = MlpNets(p.pvn3d_params.mlp_params,
        #                   num_pts=data_config.n_sample_points,
        #                   num_kpts=8,
        #                   num_cls=2,
        #                   num_cpts=1,
        #                   channel_xyz=3)

        # num_rgbd_feats = list(dense_fusion_model.output_shape)[-1]

        # mlp_model = mlp_net.build_mlp_model(rgbd_features_shape=(data_config.n_sample_points, num_rgbd_feats))

        # if p.monitor_params.weights_path is not None:
        #     pvn3d_model.load_weights(p.monitor_params.weights_path)

        warming_times = 10
        testing_times = 1000
        image_id = 288000
        training = False
        num_of_points = p.pvn3d_params.point_net2_params.n_sample_points

        # preprocessed_data_path = os.path.join(data_config.preprocessed_folder, "{:06}.bin".format(image_id))
        preprocessed_data_path = '/home/hongi/dataset/blender/blender/blender_near/01/preprocessed/{:06}.bin'.format(image_id)
        RT_gt, rgb, dpt, cam_intrinsic, crop_index, pcld_xyz, pcld_feats, sampled_index = get_data_preprocessed(
            preprocessed_data_path)

        h, w, c = data_config.rgb_input_shape
        fake_rgb = rgb[:h, :w, :]
        fake_sampled_index = np.random.randint(low=0, high=h * w, size=num_of_points)
        input_data = expand_dim(fake_rgb, pcld_xyz[:num_of_points], pcld_feats[:num_of_points], fake_sampled_index)
        rgb, pcld_xyz, pcld_feats, sampled_index = input_data
        pcld = [pcld_xyz, pcld_feats]

        # warming up:

        for _ in tqdm.tqdm(range(warming_times)):

            _, _, _ = pvn3d_forward_pass(input_data, pvn3d_model, training=False)
            _ = pointnet2_model(pcld, training=False)
            # feats = resnet_model(rgb, training=training)
            # pcld_emb = pointnet2_model(pcld, training=training)
            # rgb_features = psp_model(feats, training=training)
            # rgb_emb = match_choose(rgb_features, sampled_index)
            # feats_fused = dense_fusion_model([rgb_emb, pcld_emb], training=training)
            # kp, sm, _ = mlp_model(feats_fused, training=training)
            # if print_model_summary:
            #     print(pvn3d_model.summary())
            #     print_model_summary = False

        run_time_mean_list = []

        # with tf.profiler.experimental.Profile('logs/pointnet++_tf'):
        for i in tqdm.tqdm(range(testing_times)):
            t0 = time.perf_counter()
            # _ = pointnet2_model(pcld, training=False)
            _, _, _ = pvn3d_forward_pass(input_data, pvn3d_model, training=False)
            delta_t = time.perf_counter() - t0
            run_time_mean_list.append(delta_t)

        #
        mean = np.array(run_time_mean_list).mean()
        std = np.array(run_time_mean_list).std()
        result_mean_list.append(mean)
        result_std_list.append(std)

        print("num_samples: {}, Mean for 1000 images with rgb_shape {} is {} ".format(num_of_points, rgb_shape, mean))
        print("num_samples: {}, Std for 1000 images with rgb_shape {} is {}:".format(num_of_points, rgb_shape, std))

    # x = np.array(range(len(rgb_shape_list))) + 1
    # result_mean_list = [0.0984, 0.0990, 0.0987, 0.0931, 0.09933, 0.098]
    # result_std_list = [0.0067, 0.007, 0.00734, 0.0075, 0.0064, 0.006]
    # x = ["{}".format(shape) for shape in rgb_shape_list]

    # result_mean_list = [0.048, 0.049, 0.05, 0.048, 0.048, 0.046, 0.06, 0.072, 0.086, 0.10 ]
    # result_std_list = [0.00332, 0.0025, 0.0026, 0.0023, 0.0032, 0.0042, 0.004, 0.007, 0.007, 0.005]
    # n_samples_list = [100, 500, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 12288]
    # x = ["{}".format(n_sample) for n_sample in n_samples_list]
    # plt.bar(x, np.array(result_mean_list))
    # plt.bar(x, np.array(result_std_list))
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/pvn3d_blender_near.json', help='Path to config file')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    parser.add_argument('--id', default='demo', help='overrides the logfile name and the save name')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--save_path', default='test_plot', help="path to demo images")
    args = parser.parse_args()

    assert args.config is not None, "config is not given"

    params = read_config(args.config)
    params.monitor_params.weights_path = args.weights
    params.monitor_params.model_name = args.id
    params.monitor_params.log_file_name = args.id
    save_path = os.path.join(args.save_path, args.id)

    eval_pvn_time(params, save_path)
