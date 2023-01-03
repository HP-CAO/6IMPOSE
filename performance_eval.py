import os

import inspect
import sys
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import *


def performance_eval_linemod(p, args):
    import numpy as np
    import tqdm
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')

    try:
        tf.config.experimental.set_memory_growth(physical_devices[-1], True)
    except:
        exit("GPU allocated failed")

    from lib.net.pvn3d_adp import Pvn3dNet, forward_pass
    from lib.net.pprocessnet import InitialPoseModel
    from lib.data.utils import load_mesh, expand_dim, get_mesh_diameter, \
        rescale_image_bbox, get_crop_index, crop_image, pcld_processor_tf, formatting_predictions, \
        get_yolo_rescale_values
    from lib.data.linemod.linemod_settings import LineModSettings
    from lib.data.linemod.linemod import LineMod
    from lib.monitor.evaluator import cal_auc, cal_accuracy, cal_add_dis, cal_adds_dis, \
        get_pascalvoc_metrics, get_coco_metric

    from darknet import darknet
    tf.random.set_seed(10)

    data_config = LineModSettings(p.dataset_params.data_name,
                                  p.dataset_params.cls_type,
                                  p.dataset_params.use_preprocessed,
                                  p.dataset_params.crop_image)

    network, class_names, class_colors = darknet.load_network(
        args.yolo_config,
        args.data_file,
        args.yolo_weights,
        batch_size=1
    )

    yolo_thresh = 0.25
    test_with_gt_box = False

    width = darknet.network_width(network)
    height = darknet.network_height(network)

    yolo_rescale_factor, dw, dh = get_yolo_rescale_values()

    linemod_data_loader = LineMod(mode='train', data_name='data', cls_type=p.dataset_params.cls_type,
                                  use_preprocessed=False,
                                  size_all=10000, train_size=5000)

    resnet_w_h = 80
    resnet_input_size = [resnet_w_h, resnet_w_h]
    rgb_input_shape = [resnet_w_h, resnet_w_h, 3]
    bbox_default = [240., 160., 400., 320.]

    pvn3d_model = Pvn3dNet(p.pvn3d_params,
                           rgb_input_shape=rgb_input_shape,
                           num_kpts=data_config.n_key_points,
                           num_cls=data_config.n_classes,
                           num_cpts=data_config.n_ctr_points,
                           dim_xyz=data_config.dim_pcld_xyz)

    n_sample_points = p.pvn3d_params.point_net2_params.n_sample_points

    initial_pose_model = InitialPoseModel()

    if p.monitor_params.weights_path is not None:
        pvn3d_model.load_weights(p.monitor_params.weights_path)

    obj_id = data_config.obj_dict[data_config.cls_type]
    rescale_factor = 0.001
    mesh_path = os.path.join(data_config.mesh_dir, "obj_{:02}.ply".format(obj_id))
    mesh_points = load_mesh(mesh_path, scale=rescale_factor, n_points=500)
    mesh_info_path = os.path.join(data_config.mesh_dir, "model_info.yml")
    mesh_diameter = get_mesh_diameter(mesh_info_path, obj_id) * rescale_factor  # from mm to m
    kpts_path = os.path.join(data_config.kps_dir, "{}/farthest.txt".format(data_config.cls_type))
    corner_path = os.path.join(data_config.kps_dir, "{}/corners.txt".format(data_config.cls_type))
    key_points = np.loadtxt(kpts_path)
    center = [np.loadtxt(corner_path).mean(0)]
    mesh_kpts = np.concatenate([key_points, center], axis=0)
    mesh_kpts = tf.cast(tf.expand_dims(mesh_kpts, axis=0), dtype=tf.float32)
    intrinsic_matrix = data_config.intrinsic_matrix

    bbox2det = lambda bbox: {'coor': np.array(bbox[:4]), 'conf': np.array(bbox[4]), 'image_index': index}

    add_score_list = []
    adds_score_list = []

    gt_bboxes = []
    pred_bboxes = []

    test_index = np.loadtxt(linemod_data_loader.data_config.test_txt_path).astype(np.int)

    index = 0

    for i in tqdm.tqdm(test_index):
        index = i  # for bbox evaluation
        try:
            RT_gt_list = linemod_data_loader.get_RT_list(index=i)
            RT_gt = RT_gt_list[0][0]
        except:
            continue

        darknet_image = darknet.make_image(width, height, 3)
        image_rgb = linemod_data_loader.get_rgb(index=i)
        depth = linemod_data_loader.get_depth(index=i)
        gt_box = linemod_data_loader.get_gt_bbox(index=i)

        if gt_box is not None:
            gt_box[:, -1] = 1.0
            gt_bboxes.extend([bbox2det(box) for box in gt_box])

        image_resized = rescale_image_bbox(np.copy(image_rgb), (width, height))
        image_resized = image_resized.astype(np.uint8)

        # ===== yolo_inference =====

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=yolo_thresh)
        darknet.free_image(darknet_image)

        if len(detections) != 0:
            detect = detections[-1]  # picking the detection with highest confidence score
            bbox = formatting_predictions(detect, yolo_rescale_factor, dw, dh)
            pred_bboxes.extend([bbox2det(box) for box in [bbox]])
        else:
            bbox = bbox_default

        if test_with_gt_box:
            bbox = gt_box[0]

        crop_index, crop_factor = get_crop_index(bbox, base_crop_resolution=resnet_input_size)
        rgb = crop_image(image_rgb, crop_index)
        depth = crop_image(depth, crop_index)

        rgb_normalized = rgb.copy() / 255.

        pcld_xyz, pcld_feats, sampled_index = pcld_processor_tf(depth.astype(np.float32),
                                                                rgb_normalized.astype(np.float32), intrinsic_matrix, 1,
                                                                n_sample_points, xy_ofst=crop_index[:2],
                                                                depth_trunc=2.0)

        rgb = tf.image.resize(rgb, resnet_input_size).numpy()
        input_data = expand_dim(rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor)
        kp_pre_ofst, seg_pre, cp_pre_ofst = forward_pass(input_data, pvn3d_model, training=False)
        R, t, _ = initial_pose_model([input_data[1], kp_pre_ofst, cp_pre_ofst, seg_pre, mesh_kpts], training=False)
        Rt_pre = np.zeros((3, 4))

        Rt_pre[:, :3] = R[0]
        Rt_pre[:, 3] = t[0]

        add_score = cal_add_dis(mesh_points, Rt_pre, RT_gt)
        add_score_list.append(add_score)
        adds_score = cal_adds_dis(mesh_points, Rt_pre, RT_gt)
        adds_score_list.append(adds_score)

    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = np.array(pred_bboxes)
    ap_50 = get_pascalvoc_metrics(gt_bboxes, pred_bboxes)
    ap_75 = get_pascalvoc_metrics(gt_bboxes, pred_bboxes, iou_threshold=0.75)
    ap_coco = get_coco_metric(gt_bboxes, pred_bboxes)

    bbox_result = [{'name': 'AP@0.5', 'type': 'scalar', 'data': ap_50},
                   {'name': 'AP@0.75', 'type': 'scalar', 'data': ap_75},
                   {'name': 'AP (COCO)', 'type': 'scalar', 'data': ap_coco}]

    print("bbox result:\n", bbox_result)

    add_auc = cal_auc(add_score_list, max_dis=0.1)
    add_mean = np.mean(add_score_list)
    add_accuracy = cal_accuracy(add_score_list, dis_threshold=0.1 * mesh_diameter)
    adds_auc = cal_auc(adds_score_list, max_dis=0.1)
    adds_accuracy = cal_accuracy(adds_score_list, dis_threshold=0.1 * mesh_diameter)
    print("Without icp == add_mean:{},  add_auc: {} , adds_auc: {} , add_acc: {}, adds_acc: {}".
          format(add_mean, add_auc, adds_auc, add_accuracy, adds_accuracy))

    # result_save_path = os.path.join("paper_script/pose_result", data_config.cls_type)
    # print("the evaluation result saved")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/pvn3d_test.json', help='Path to config file')
    parser.add_argument('--id', default='demo', help='overrides the logfile name and the save name')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--save_path', default='test_plot', help="path to demo images")
    parser.add_argument('--weights', default='models/sim2real_duck_8_best/pvn3d', help='Path to pretrained weights')
    parser.add_argument('--gpu_id', default="0")

    parser.add_argument('--yolo_config', default="config/yolo_config/yolov4-tiny-lm-all.cfg",
                        help="path to config file")
    parser.add_argument('--yolo_weights', default="models/yolo_weights/yolov4-tiny-lm-all_best.weights",
                        help="yolo weights path")
    parser.add_argument('--data_file', default="./config/yolo_config/single_obj.data",
                        help="path to data file")

    args = parser.parse_args()

    assert args.config is not None, "config is not given"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    params = read_config(args.config)
    params.monitor_params.weights_path = args.weights
    params.monitor_params.model_name = args.id
    params.monitor_params.log_file_name = args.id
    save_path = os.path.join(args.save_path, args.id)

    performance_eval_linemod(params, args)
