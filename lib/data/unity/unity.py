import os
import random
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from lib.data.unity.unity_settings import UnitySettings
from lib.utils import CachedProperty
from lib.data.utils import convert2pascal


class Unity:
    def __init__(self, mode, data_name, cls_type, use_preprocessed, size_all, train_size,
                 crop_image=False, shuffle=True, add_noise=True):

        self.dataset_name = data_name
        self.mode = mode.decode("utf-8") if isinstance(mode, bytes) else mode
        self.cls_type = cls_type.decode("utf-8") if isinstance(cls_type, bytes) else cls_type
        self.crop_image = crop_image
        self.counter = 0
        self.shuffle = shuffle
        self.demo_size = 10
        self.add_noise = add_noise

        # rgb is in [0, 1]
        self.p_sat = 0.5
        self.p_bright = 0.5
        self.p_noise = 0.1
        self.p_hue = 0.03
        self.p_contr = 0.5
        self.dp_noise = 0.01

        self.data_config = \
            UnitySettings(data_name, self.cls_type, use_preprocessed=use_preprocessed,
                          crop_image=crop_image, size_all=size_all, train_size=train_size)

        self.cls_id = self.data_config.obj_dict[self.cls_type]
        self.cls_root = os.path.join(self.data_config.root, "%02d" % self.cls_id)
        self.obj_cls_id = 1 if cls_type != "all" else self.cls_id


    def get_rgb(self, index, rgb_path=None):

        if rgb_path is None:
            rgb_path = os.path.join(self.cls_root, "rgb/{}.png".format(str(index).rjust(4, "0")))

        with Image.open(rgb_path) as rgb:
            rgb = np.array(rgb).astype(float)
            if self.add_noise and self.mode == 'train':
                rgb = self.augment_rgb(rgb) # TODO replace with augmenter
            return rgb

    def get_depth(self, index):
        with Image.open(os.path.join(self.cls_root, "depth/{}.png".format(str(index).rjust(4, "0")))) as depth:
            dpt = np.array(depth)
            if self.add_noise and self.mode == 'train':
                dpt = self.augment_depth(dpt) # TODO replace with augmenter
            return dpt

    def augment_rgb(self, rgb):
        if self.p_sat > 0:
            rgb = tf.image.random_saturation(rgb, 1. - self.p_sat, 1. + self.p_sat)
        if self.p_hue > 0:
            rgb = tf.image.random_hue(rgb, self.p_hue)
        if self.p_contr > 0:
            rgb = tf.image.random_contrast(rgb, 1. - self.p_contr, 1. + self.p_contr)
        if self.p_bright > 0:
            rgb = tf.image.random_brightness(rgb, self.p_bright)
        if self.p_noise > 0:
            noise = tf.random.normal(shape=tf.shape(rgb), mean=0.0, stddev=self.p_noise, dtype=tf.float32)
            rgb = tf.cast(rgb, tf.float32) + tf.cast(noise, tf.float32)
        return rgb.numpy()

    def augment_depth(self, depth):

        if self.p_noise > 0:
            # [-0.04, 0.04]
            noise = tf.random.normal(shape=tf.shape(depth), mean=0.0, stddev=self.dp_noise, dtype=tf.float32)
            depth = tf.cast(depth, tf.float32) + tf.cast(noise, tf.float32)
        return depth.numpy()

    def get_gt_bbox(self, index):
        """
        return:
            gt_bbox, shape: [n_boxes, 5], where for each bounding_box, returns a list of [x, y, h, w, id]
        """
        meta_list = []
        bbox_list = []
        meta = self.meta_lst[index]

        if self.cls_id == 2:
            for i in range(0, len(meta)):
                if meta[i]['obj_id'] == 2:
                    meta_list.append(meta[i])
                    break
        elif self.cls_id == 16:
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

    def get_item(self):
        """
        this function is used to get the raw rgb and raw depth from the LineMod dataset
        this function will be overridden when call its children class LineModPvn or LineModYolo
        """
        index = self.index_lst[self.counter]
        rgb = self.get_rgb(index)
        depth = self.get_depth(index)
        return rgb, depth

    @CachedProperty
    def meta_lst(self):
        with open(os.path.join(self.cls_root, 'gt.yml'), "r") as meta_file:
            return yaml.load(meta_file, Loader=yaml.FullLoader)

    @CachedProperty
    def size_all(self):

        size = len(self.meta_lst)
        if self.data_config.use_preprocessed:
            size = len(os.listdir(self.data_config.preprocessed_folder))
        # print("Training on {} dataset: {} images".format(self.dataset_name, size))
        return size

    @CachedProperty
    def index_lst(self):

        if self.mode == 'train':
            index_lst = range(0, self.data_config.train_size)
        elif self.mode == 'val':
            index_lst = range(self.data_config.train_size, self.data_config.size_all)
        else:
            # test from validation dataset
            index_lst = range(self.data_config.train_size, self.data_config.size_all)
            # index_lst = range(0, train_size)
            index_lst = random.sample(index_lst, self.demo_size)
            return index_lst

        if self.shuffle:
            index_lst = random.sample(index_lst, len(index_lst))
        return index_lst

    def has_next(self):
        return self.counter < (len(self.index_lst) - 1)

    def next(self):
        item = self.get_item()
        self.counter += 1
        return item

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            raise StopIteration()


def get_depth_value(img_dpt, clip_range=(1.0, 2.5)):
    """
    convert depth image(in rgb representation) to depth value
    params: img_dpt: depth image generated from unity [W, H, 3]

            clip_range: depth map values lies in the clip range,
            typically pixel value 0 means black in greyscale depth_image, meaning the nearest distance
            for example

    return: depth map with values in meters [W, H]
    """
    clip_range_l, clip_range_u = clip_range
    dpt_map = (img_dpt[:, :, 0] / 255) * (clip_range_u - clip_range_l) + clip_range_l
    return dpt_map


def unity_down_sample(image, downsample_factor=6, method='nearest'):
    """
    down_sampling the origin 4K image by downsample_factor
    params: image: [W, H, 3]   4K: 2160 * 3840
    return: a resized image of a numpy array
    """
    if len(image.shape) == 2:
        h, w = image.shape
        image = tf.expand_dims(image, axis=2)
        image = tf.repeat(image, repeats=3, axis=2)
    else:
        h, w, _ = image.shape

    h_new = int(h / downsample_factor)
    w_new = int(w / downsample_factor)

    h_new = tf.constant(h_new, dtype=tf.int32)
    w_new = tf.constant(w_new, dtype=tf.int32)

    img_resized = tf.image.resize(image, size=(h_new, w_new), method=method)

    return img_resized.numpy()
