import random
import cv2
import os
import numpy as np
from lib.monitor.monitor import MonitorParams
from lib.network import Network
from lib.params import DatasetParams, NetworkParams, Networks
from lib.trainer.trainer import TrainerParams
from darknet import darknet
from darknet.darknet import bbox2points
from lib.data.utils import rescale_image_bbox
from subprocess import Popen, TimeoutExpired
import signal
import matplotlib.pyplot as plt

from utils import ensure_fd


class DarknetParams:
    cfg_file: str


class MainDarknetParams(NetworkParams):
    def __init__(self):
        self.network = Networks.pvn3d
        self.dataset_params = DatasetParams()
        self.monitor_params = MonitorParams()
        self.trainer_params = TrainerParams()
        self.darknet_params = DarknetParams()


class MainDarknet(Network):
    params: MainDarknetParams

    def __init__(self, params: MainDarknetParams):
        super().__init__(params)
        self.dn_binary = './darknet/darknet'

        self.network = None
        self.cfg_file = self.params.darknet_params.cfg_file
        ds_folder = os.path.join(self.data_config.preprocessed_folder, 'darknet')
        self.data_file = os.path.join(ds_folder, 'obj.data')

        self.batch_size = self.global_train_batch_size
        self.conf_thresh = 0.3

    def train(self):
        """ overwrite tf training """

        ensure_fd(os.path.join(self.params.monitor_params.model_dir, self.params.monitor_params.model_name))

        gpu_list = self.params.trainer_params.distribute_train_device

        train_cmd = [self.dn_binary, "detector", "train", self.data_file, self.cfg_file, "-gpus", str(gpu_list),
                     "-mjpeg_port", "8090", "-map", "-dont_show"]

        proc = Popen(train_cmd, shell=False, preexec_fn=os.setsid)
        try:
            proc.wait()
        except (KeyboardInterrupt, TimeoutExpired):
            os.killpg(proc.pid, signal.SIGINT)
            print("Terminating...")

    def initial_trainer_and_model(self):
        random.seed(3)  # deterministic bbox colors
        self.network, self.class_names, self.class_colors = darknet.load_network(
            self.cfg_file,
            self.data_file,
            self.params.monitor_params.weights_path,
            self.batch_size)

        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.image_buffer = darknet.make_image(self.width, self.height, 3)

    def run_demo(self):
        """ run detection on provided dataset """
        self.initial_trainer_and_model()

        from lib.factory import DatasetFactory
        factory = DatasetFactory(self.params)
        dataset = factory.get_dataset('val')
        dataset.data_config.use_preprocessed = False

        def on_press(event):
            rgb, gt_bboxes = dataset.next()
            bgr, detections, gt_bboxes = self.image_detection(rgb, self.conf_thresh, gt_bboxes)
            darknet.print_detections(detections, True)
            plt.imshow(bgr)
            fig.canvas.draw()

        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', on_press)

        on_press(None)
        plt.show()

    def performance_evaluation(self, epoch):
        """ evaluate on dataset """

        weights_path = self.params.monitor_params.weights_path

        iou_threshold = 0.5

        eval_cmd = [self.dn_binary, "detector", "map", self.data_file, self.cfg_file, weights_path, "-iou_thresh",
                    str(iou_threshold)]

        proc = Popen(eval_cmd, shell=False, preexec_fn=os.setsid)
        try:
            proc.wait()
        except (KeyboardInterrupt, TimeoutExpired):
            os.killpg(proc.pid, signal.SIGINT)
            print("Terminating...")

    def yolo_bbox_2_original(self, bbox_xywh, original_rgb_shape):
        bbox = [(bbox_xywh[0] - bbox_xywh[2] / 2),
                (bbox_xywh[1] - bbox_xywh[3] / 2),
                (bbox_xywh[0] + bbox_xywh[2] / 2),
                (bbox_xywh[1] + bbox_xywh[3] / 2)]

        ih, iw = original_rgb_shape
        scale = max(iw / self.width, ih / self.height)

        nw, nh = int(scale * self.width), int(scale * self.height)

        dw, dh = (iw - nw) // 2, (ih - nh) // 2

        bbox[0] = int(bbox[0] * scale + dw)
        bbox[1] = int(bbox[1] * scale + dh)
        bbox[2] = int(bbox[2] * scale + dw)
        bbox[3] = int(bbox[3] * scale + dh)

        return bbox

    def check_batch_shape(self, images, batch_size):
        """
            Image sizes should be the same width and height
        """
        shapes = [image.shape for image in images]
        if len(set(shapes)) > 1:
            raise ValueError("Images don't have same shape")
        if len(shapes) > batch_size:
            raise ValueError("Batch size higher than number of images")
        return shapes[0]

    def prepare_batch(self, images, channels=3):
        width = darknet.network_width(self.network)
        height = darknet.network_height(self.network)

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

    def image_detection(self, image_or_path, thresh, gt_box):
        if type(image_or_path) is str:
            image = cv2.imread(image_or_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_or_path

        if gt_box is not None:
            image_resized, gt_boxes = rescale_image_bbox(image_rgb, (self.width, self.height), gt_box)
        else:
            image_resized = rescale_image_bbox(image_rgb, (self.width, self.height), None)
            gt_boxes = None

        image_resized = image_resized.astype(np.uint8)

        darknet.copy_image_from_bytes(self.image_buffer, image_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, self.image_buffer, thresh=thresh)

        # darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, self.class_colors)

        return image, detections, gt_boxes

    def batch_detection(self, images, class_names, class_colors,
                        thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
        image_height, image_width, _ = self.check_batch_shape(images, batch_size)
        darknet_images = self.prepare_batch(images)
        batch_detections = darknet.network_predict_batch(self.network, darknet_images, batch_size, image_width,
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

    def image_classification(self, image, class_names):
        width = darknet.network_width(self.network)
        height = darknet.network_height(self.network)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.predict_image(self.network, darknet_image)
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

    def formatting_predictions(image, detections):
        bbox_list = []
        # original_w = 640.
        # origianl_h = 480.
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            xmin = x - w / 2
            ymin = y - h / 2
            xmax = x + w / 2
            ymax = y + h / 2

            if label == 'duck':  # todo here to add all objects
                bbox = [xmin, ymin, xmax, ymax, float(confidence)]
                bbox_list.append(bbox)

        return bbox_list

    # def batch_detection_example():
    #     args = parser()
    #     check_arguments_errors(args)
    #     batch_size = 3
    #     random.seed(3)  # deterministic bbox colors
    #     network, class_names, class_colors = darknet.load_network(
    #         args.config_file,
    #         args.data_file,
    #         args.weights,
    #         batch_size=batch_size
    #     )
    #     image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    #     images = [cv2.imread(image) for image in image_names]
    #     images, detections, = batch_detection(network, images, class_names,
    #                                           class_colors, batch_size=batch_size)
    #     for name, image in zip(image_names, images):
    #         cv2.imwrite(name.replace("data/", ""), image)
    #     print(detections)

    def pre_training(self):
        pass

    def train_step(self, inputs):
        pass

    def val_step(self, inputs):
        pass

    def forward_pass(self, input_data, training=False):
        pass

    def export_model(self):
        pass
