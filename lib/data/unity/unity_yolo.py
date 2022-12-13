import numpy as np
from lib.data.unity.unity import Unity
from lib.data.utils import rescale_image_bbox
from lib.data.augmenter import random_horizontal_flip, random_crop, random_translate
from lib.net.utils import bbox_iou
import os
import pickle
from PIL import Image


class UnityYolo(Unity):
    def __init__(self, mode, data_name, cls_type, use_preprocessed, strides, anchors, size_all,
                 train_size, shuffle=True, add_noise=True):
        super().__init__(mode, data_name, cls_type, use_preprocessed, size_all, train_size,
                         crop_image=False, shuffle=shuffle, add_noise=add_noise)

        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 100
        self.strides = strides
        self.anchors = anchors

        self.train_output_h = self.data_config.yolo_default_rgb_h // self.strides
        self.train_output_w = self.data_config.yolo_default_rgb_w // self.strides

        self.downsample_factor = 1
        self.num_classes = self.data_config.n_classes

    def get_item(self):
        index = self.index_lst[self.counter]

        if self.data_config.use_preprocessed:
            data_path = os.path.join(self.data_config.preprocessed_folder, "train_val/{:06}.bin".format(index))
            meta_data = pickle.load(open(data_path, "rb"))

            rgb_rescaled = meta_data['rgb_rescaled']
            label_sbbox = meta_data['label_sbbox']
            label_mbbox = meta_data['label_mbbox']
            label_lbbox = meta_data['label_lbbox']
            sbboxes = meta_data['sbboxes']
            mbboxes = meta_data['mbboxes']
            lbboxes = meta_data['lbboxes']

            if self.add_noise and self.mode == 'train':
                rgb_rescaled = self.augment_rgb(rgb_rescaled)

            return rgb_rescaled, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

        else:
            return

    @staticmethod
    def spatial_augmentation(image, gt_bbox):
        image, gt_bbox = random_horizontal_flip(np.copy(image), np.copy(gt_bbox))
        image, gt_bbox = random_crop(np.copy(image), np.copy(gt_bbox))
        image, gt_bbox = random_translate(np.copy(image), np.copy(gt_bbox))
        return image, gt_bbox

    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.train_output_h[i], self.train_output_w[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]

        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]

        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)

            if self.num_classes == 2:
                bbox_class_ind = 1

            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False

            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)

                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
