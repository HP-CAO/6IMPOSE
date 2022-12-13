import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import yaml
import tensorflow as tf

from darknet import darknet


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="./dataset/linemod/linemod/data/12/",
                        help="image source. It can be a single image, a"
                             "txt with paths to them, or a folder. Image valid"
                             " formats are jpg, jpeg or png."
                             "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="models/yolo_weights/yolov4-tiny-lm-all_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./config/yolo_config/yolov4-tiny-lm-all.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./dataset/darknet_yolo/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    parser.add_argument("--cls_id", type=int, default=12,
                        help="object id")
    parser.add_argument("--run_id", type=str, default="yolo-test",
                        help="run id")
    parser.add_argument("--gpu_id", default="0",
                        help="gpu id")

    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise (ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise (ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise (ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise (ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
               glob.glob(os.path.join(images_path, "*.png")) + \
               glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32) / 255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_or_path, network, class_names, class_colors, thresh, gt_box):
    from lib.data.utils import rescale_image_bbox
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    t0 = time.time()
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    if type(image_or_path) is str:
        image = cv2.imread(image_or_path)
    else:
        image = image_or_path

    # image_rgb = image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized, gt_boxes = rescale_image_bbox(image_rgb, (width, height), gt_box)
    image_resized = image_resized.astype(np.uint8)

    # image_resized = image_rgb # todo this is for original size 640 x 480
    # gt_boxes = gt_box

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

    darknet.free_image(darknet_image)
    # image = darknet.draw_boxes(detections, image_resized, class_colors)
    # todo only draw the best one
    if len(detections) != 0:
        image = darknet.draw_boxes([detections[-1]], image_resized, class_colors)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections, gt_boxes


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x / width, y / height, w / width, h / height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def formatting_predictions(image, detections, cls_type):
    bbox_list = []
    # original_w = 640.
    # origianl_h = 480.
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        # x, y, w, h = convert2relative(image, bbox)
        # x *= original_w
        # y *= origianl_h
        # w *= original_w
        # h *= origianl_h

        xmin = x - w/2
        ymin = y - h/2
        xmax = x + w/2
        ymax = y + h/2

        if label == cls_type:  # todo here to add all objects
            bbox = [xmin, ymin, xmax, ymax, float(confidence)]
            bbox_list.append(bbox)

    return bbox_list


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections, = batch_detection(network, images, class_names,
                                          class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


def get_gt_bbox(meta_lst, index, cls_id):
    from lib.data.utils import convert2pascal
    meta_list = []
    bbox_list = []
    meta = meta_lst[index]

    if cls_id == 2:
        for i in range(0, len(meta)):
            if meta[i]['obj_id'] == 2:
                meta_list.append(meta[i])
                break
    elif cls_id == 16:
        for i in range(0, len(meta)):
            meta_list.append(meta[i])
    else:
        meta_list.append(meta[0])

    for mt in meta_list:
        bbox = np.array(mt['obj_bb'])
        bbox_id = mt['obj_id']
        bbox = convert2pascal(bbox)
        bbox = np.append(bbox, bbox_id)
        bbox_list.append(bbox)
    return np.array(bbox_list)


def main():
    number = 0
    args = parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[-1], True)
    except:
        exit("GPU allocated failed")

    from lib.monitor.evaluator import get_pascalvoc_metrics, get_coco_metric

    check_arguments_errors(args)
    bbox2det = lambda bbox: {'coor': np.array(bbox[:4]), 'conf': np.array(bbox[4]), 'image_index': index}

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
    rgb_path = os.path.join(args.input, 'rgb')
    gt_meta_path = os.path.join(args.input, 'gt.yml')

    with open(gt_meta_path, "r") as meta_file:
        meta_list = yaml.load(meta_file, Loader=yaml.FullLoader)

    images = load_images(rgb_path)

    index = 0
    gt_bboxes = []
    pred_bboxes = []

    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_name = os.path.join(rgb_path, '{:04}.png'.format(index))
        else:
            image_name = input("Enter Image Path: ")

        prev_time = time.time()
        gt_box = get_gt_bbox(meta_list, index, cls_id=args.cls_id)

        image, detections, gt_box = image_detection(image_name, network, class_names, class_colors, args.thresh, gt_box)

        if args.save_labels:
            save_annotations(image_name, image, detections, class_names)
        darknet.print_detections(detections, args.ext_output)

        if gt_box is not None:
            gt_box[:, -1] = 1.0
            gt_bboxes.extend([bbox2det(box) for box in gt_box])

        if len(detections) != 0:
            pre_box = formatting_predictions(image, detections, cls_type='target')  #here object is for binary prediction
            pred_bboxes.extend([bbox2det(box) for box in pre_box])
        else:
            # print(f"Index:{index}")
            number += 1

        # fps = int(1 / (time.time() - prev_time))
        # print("FPS: {}".format(fps))
        if not args.dont_show:
            cv2.imshow('Inference', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        index += 1

    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = np.array(pred_bboxes)
    ap_50 = get_pascalvoc_metrics(gt_bboxes, pred_bboxes)
    ap_75 = get_pascalvoc_metrics(gt_bboxes, pred_bboxes, iou_threshold=0.75)
    ap_coco = get_coco_metric(gt_bboxes, pred_bboxes)

    result = [{'name': 'AP@0.5', 'type': 'scalar', 'data': ap_50},
              {'name': 'AP@0.75', 'type': 'scalar', 'data': ap_75},
              {'name': 'AP (COCO)', 'type': 'scalar', 'data': ap_coco}]

    print(result)
    print(f"{number} objects missed")
    result_save_path = os.path.join("paper_script/yolo_result", args.run_id)
    result = np.array([args.run_id, number, ap_50, ap_75, ap_coco])
    np.save(result_save_path, result)


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
