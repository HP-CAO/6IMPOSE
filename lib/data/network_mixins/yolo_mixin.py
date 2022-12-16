import os
import numpy as np
from lib.data.augmenter import augment_rgb, rotate_datapoint, random_crop
from lib.data.dataset import NoMaskError
from lib.data.utils import rescale_image_bbox, validate_bbox, get_bbox_from_mask
from lib.net.utils import bbox_iou


class YoloMixin():
    def __init__(self, strides, anchors, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 100
        self.strides = strides
        self.anchors = anchors

        if strides is not None:
            self.train_output_h = self.data_config.yolo_default_rgb_h // self.strides
            self.train_output_w = self.data_config.yolo_default_rgb_w // self.strides

        self.downsample_factor = 1
        self.num_classes = self.data_config.n_classes

    def get(self, index):
        if self.data_config.use_preprocessed:
            get_data = lambda name: np.load(os.path.join(self.data_config.preprocessed_folder, name, f"{index:06}.npy"))
            rgb_rescaled = get_data('yolo_rgb_input')
            real_bbox = get_data('gt_bboxes')
            return rgb_rescaled, real_bbox

        else:
            data = self.get_dict(index)
            rgb_rescaled = data['yolo_rgb_input']
            gt_bboxes = data['gt_bboxes']
            return rgb_rescaled, gt_bboxes

    def get_dict(self, index):
        rgb = self.get_rgb(index)

        if self.if_augment:
            rgb = augment_rgb(rgb.astype(np.float32) / 255.) * 255

        try:
            mask = self.get_mask(index)
            if self.data_config.augment_per_image > 1:
                rgb, mask = rotate_datapoint(img_likes=[rgb, mask])

            # we use mask for bbox to get exact bbox for all rotations
            bboxes = []
            if self.cls_type == 'all':
                for cls, gt_mask_value in self.data_config.mask_ids.items():
                    bbox = get_bbox_from_mask(mask, gt_mask_value)
                    if bbox is None:
                        continue
                    bbox = list(bbox)
                    bbox.append(self.data_config.obj_dict[cls])
                    bboxes.append(bbox)
            else:
                bbox = get_bbox_from_mask(mask, gt_mask_value=255)
                if bbox is not None:
                    bbox = list(bbox)
                    bbox.append(0)
                    bboxes.append(bbox)

        except NoMaskError:
            bboxes = self.get_gt_bbox(index)

        # filter too small bboxes
        bboxes = [bbox for bbox in bboxes if validate_bbox(bbox)]
        bboxes = np.array(bboxes)

        if self.if_augment:
            if len(bboxes) > 0:
                rgb, bboxes = random_crop(rgb, bboxes)

        yolo_default_rgb_h = self.data_config.yolo_default_rgb_h
        yolo_default_rgb_w = self.data_config.yolo_default_rgb_w

        rgb_rescaled, gt_bboxes \
            = rescale_image_bbox(image=rgb, target_size=[yolo_default_rgb_h, yolo_default_rgb_w], gt_boxes=bboxes)

        # label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(gt_bboxes)

        data = {}
        data['yolo_rgb_input'] = rgb_rescaled.astype(np.uint8)
        # data['label_sbbox'] = label_sbbox.astype(np.float32)
        # data['label_mbbox'] = label_mbbox.astype(np.float32)
        # data['label_lbbox'] = label_lbbox.astype(np.float32)
        # data['sbboxes'] = sbboxes.astype(np.float32)
        # data['mbboxes'] = mbboxes.astype(np.float32)
        # data['lbboxes'] = lbboxes.astype(np.float32)
        data['gt_bboxes'] = gt_bboxes.astype(np.float32)

        return data

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
