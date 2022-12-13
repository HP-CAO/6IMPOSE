import tensorflow as tf
from lib.params import DatasetParams


class PvnDatasetParams(DatasetParams):
    def __init__(self):
        self.dataset = 'unity'
        self.data_name = "img_crop"
        self.cls_type = "ape"
        self.train_batch_size = 8
        self.val_batch_size = 16
        self.test_batch_size = 1
        self.crop_image = True
        self.img_downsample_factor = 1
        self.use_preprocessed = False
        self.use_pvn_kp = False
        self.train_size = 69930
        self.size_all = 71190
        self.augment_per_image = 0


class YoloDatasetParams(DatasetParams):
    def __init__(self):
        self.dataset = 'blender'
        self.data_name = 'blender_near'
        self.cls_type = 'duck'
        self.use_preprocessed = False
        self.train_batch_size = 8
        self.val_batch_size = 16
        self.test_batch_size = 1
        self.train_size = 69930
        self.size_all = 71190
        self.test_num = 10


def normal_tensor_spec_test():

    tensor_spec = (
        tf.TensorSpec(shape=(480, 640, 3), dtype=tf.float32, name='rgb'),
        tf.TensorSpec(shape=(480, 640), dtype=tf.float32, name='depth'),
    )
    return tensor_spec


def pvn3d_tensor_spec(cls_type, rgb_input_shape, n_points, mode="test", use_pvn_kp=False):

    label_shape = (n_points, 10) if cls_type == 'all' else (n_points, 2)

    if mode == "test":

        tensor_spec_list = [
            tf.TensorSpec(shape=rgb_input_shape, dtype=tf.float32, name='rgb'),
            tf.TensorSpec(shape=(n_points, 3), dtype=tf.float32, name='pcld_xyz'),
            tf.TensorSpec(shape=(n_points, 6), dtype=tf.float32, name='pcld_feats'),
            tf.TensorSpec(shape=n_points, dtype=tf.int32, name='sampled_index'),
            tf.TensorSpec(shape=1, dtype=tf.int8, name='crop_factor')
        ]
        if use_pvn_kp:
            tensor_spec_list.append(
                tf.TensorSpec(shape=1, dtype=tf.int8, name='cls_id')
            )
    else:
        tensor_spec_list = [
            tf.TensorSpec(shape=rgb_input_shape, dtype=tf.uint8, name='rgb'),
            tf.TensorSpec(shape=(n_points, 3), dtype=tf.float32, name='pcld_xyz'),
            tf.TensorSpec(shape=(n_points, 6), dtype=tf.float32, name='pcld_feats'),
            tf.TensorSpec(shape=n_points, dtype=tf.int32, name='sampled_index'),
            tf.TensorSpec(shape=label_shape, dtype=tf.int8, name='labels'),
            tf.TensorSpec(shape=(n_points, 8, 3), dtype=tf.float32, name='kpts_targ_offst'),
            tf.TensorSpec(shape=(n_points, 1, 3), dtype=tf.float32, name='ctr_targ_offst'),
            tf.TensorSpec(shape=n_points, dtype=tf.float32, name='mask_label'),
            tf.TensorSpec(shape=(), dtype=tf.int8, name='crop_factor'),
        ]
        if use_pvn_kp:
            tensor_spec_list.append(tf.TensorSpec(shape=1, dtype=tf.int8, name='cls_id'))
            tensor_spec_list.append(tf.TensorSpec(shape=(9, 3), dtype=tf.float32, name='kp_cp_target'))

    tensor_spec_tuple = tuple(tensor_spec_list)

    return tensor_spec_tuple


def yolo_tensor_spec(num_cls, yolo_rgb_shape, rgb_shape=None, mode="test"):

    if mode == "test":
        tensor_spec = (
            tf.TensorSpec(shape=rgb_shape, dtype=tf.float32, name='rgb_input')
        )

    else:
        tensor_spec = (
            tf.TensorSpec(shape=yolo_rgb_shape, dtype=tf.uint8, name='yolo_rgb_input'),
            tf.TensorSpec(shape=(26, 26, 3, 5 + num_cls), dtype=tf.float32, name='label_sbbox'),
            tf.TensorSpec(shape=(13, 13, 3, 5 + num_cls), dtype=tf.float32, name='label_mbbox'),
            tf.TensorSpec(shape=(6, 6, 3, 5 + num_cls), dtype=tf.float32, name='label_lbbox'),
            tf.TensorSpec(shape=(100, 4), dtype=tf.float32, name='sbboxes'),
            tf.TensorSpec(shape=(100, 4), dtype=tf.float32, name='mbboxes'),
            tf.TensorSpec(shape=(100, 4), dtype=tf.float32, name='lbboxes')
        )

    return tensor_spec
