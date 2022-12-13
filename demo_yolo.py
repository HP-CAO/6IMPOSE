import os
import cv2
import numpy as np
from PIL import Image
import argparse
import pickle
from lib.net.yolov3 import YoloV3
from lib.monitor.evaluator import detect_image
from lib.monitor.visualizer import draw_bbox
from lib.data.blender.blender_settings import BlenderSettings
from utils import *
import shelve
import time


def demo_single_image(params):
    data_config = BlenderSettings(params.dataset_params.data_name, params.dataset_params.cls_type,
                                  params.dataset_params.use_data_preprocessed, False, params.dataset_params.size_all,
                                  params.dataset_params.train_size)

    low_value = params.dataset_params.size_all - params.dataset_params.train_size
    random_picks = np.random.randint(low=low_value, high=params.dataset_params.size_all, size=100)
    save_folder = "./yolo_demo_result"
    ensure_fd(save_folder)

    yolo_net = YoloV3(params.yolo_params,
                      num_cls=data_config.n_classes,
                      yolo_input_shape=data_config.yolo_rgb_shape,
                      ori_image_shape=data_config.ori_rgb_shape)

    yolo_model = yolo_net.build_yolo_tiny_model(mode=params.monitor_params.mode)  # bgr [0, 1] image input

    intrinsic_matrix = np.array([[1.347360229492187500e+03, 0.000000000000000000e+00, 9.837904052734375000e+02],
                                 [0.000000000000000000e+00, 1.347594360351562500e+03, 5.217407836914062500e+02],
                                 [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]]
                                )

    frame_shape = (1080, 1920)

    if params.monitor_params.weights_path is not None:
        yolo_model.load_weights(params.monitor_params.weights_path)
        print('Pre-trained model loaded successfully')

    with shelve.open(f"camera/test_video") as d:
        times = d["times"]
        frames = d["frames"]
        dts = d["dts"]

    actual_t_start = time.perf_counter()

    for frame, dt, t_frame in zip(frames, dts, times):

        # print(f"t_frame: {t_frame:.2f}, t_actual: {time.perf_counter()-actual_t_start:.2f}")

        t_infer = time.perf_counter()
        bboxes, bgr_image = detect_image(yolo_model, rgb_image=frame.rgb,
                                         input_size=data_config.yolo_default_rgb_h,
                                         cls_type=params.dataset_params.cls_type)

        bgr_image = draw_bbox(bgr_image, bboxes, params.dataset_params.cls_type, rectangle_colors=[0, 0, 255])
        t_infer = time.perf_counter() - t_infer
        print("infer_time", t_infer)

        cv2.imshow("CAM rgb", cv2.resize(bgr_image, (640, 480)))

        cv2.waitKey(1)

    print("average FPS: ", len(frames) / (time.perf_counter() - actual_t_start))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', action='store_true', help='Activate usage of GPU')
    parser.add_argument('--gpu', default=True, help='Activate usage of GPU')
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default='config/yolo_blender.json', help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--force', action='store_true', help='Override log file without asking')
    parser.add_argument('--weights', default='models/yolo_blender/yolo', help='Path to pretrained weights')
    # './model/model_name' or path to .blob
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--mode', default='test', help='Choose the mode, train or test')

    args = parser.parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    if args.config is None:
        exit("config file needed")

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    if args.id is not None:
        params.monitor_params.model_name = args.id
        params.monitor_params.log_file_name = args.id

    if args.force:
        params.monitor_params.force_override = True

    if args.weights is not None:
        params.monitor_params.weights_path = args.weights
    else:
        exit("Error: Please load pre-trained model first")

    params.monitor_params.mode = args.mode

    demo_single_image(params)
