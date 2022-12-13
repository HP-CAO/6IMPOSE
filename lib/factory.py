from builtins import AssertionError
from lib.data.dataset import Dataset
from lib.data.dataset_settings import DatasetSettings
from lib.network import Network
from lib.params import NetworkParams, Networks, Datasets
import numpy as np
import tensorflow as tf
import os


class NetworkFactory:
    params: NetworkParams

    def __init__(self, params: NetworkParams = None):
        self.params = params

    def get_network(self) -> Network:
        """ build network according to params"""
        if self.params.network == Networks.pvn3d:
            from lib.main_pvn import MainPvn3d
            return MainPvn3d(self.params)
        elif self.params.network == Networks.darknet:
            from lib.main_darknet import MainDarknet
            return MainDarknet(self.params)
        else:
            raise AssertionError(f"Unknown network: {self.params.network}")

    def get_default_params(self) -> NetworkParams:
        """ get default params """
        if self.params.network == Networks.yolo:
            raise NotImplementedError("We use Darknet implementation now!")
        elif self.params.network == Networks.pvn3d:
            from lib.main_pvn import MainPvn3dParams
            return MainPvn3dParams()
        else:
            raise AssertionError(f"Unknown network: {self.params.network}")


class DatasetFactory:
    params: NetworkParams

    def __init__(self, params: NetworkParams = None):
        self.params = params

    def build_dataset(self, dataset: str, data_name: str, cls_type: str, **kwargs):
        """ from args not from params """
        args = ('train', data_name, cls_type)

        if dataset == Datasets.blender:
            from lib.data.blender.blender import Blender
            gen = Blender
        elif dataset == Datasets.linemod:
            from lib.data.linemod.linemod import LineMod
            gen = LineMod
        elif dataset == Datasets.unity_grocieries_real:
            from lib.data.unity_groceries_real.unity_groceries_real import UGReal
            gen = UGReal
        else:
            raise AssertionError("Unknown dataset: ", dataset)

        return gen(*args, **kwargs)

    def get_data_config(self) -> DatasetSettings:
        _, _, _, data_config = self.__get_gen_args_spec_dataconfig('train')  # mode does not matter here
        return data_config

    def get_len(self, mode) -> int:
        ds = self.get_dataset(mode)
        return len(ds)

    def get_dataset(self, mode) -> Dataset:
        """ returns 'Dataset' according to params and mode = {'train', 'val', 'preprocess'}"""
        gen, args, _, _ = self.__get_gen_args_spec_dataconfig(mode)
        return gen(*args)

    def data_from_generator(self, mode) -> tf.data.Dataset:
        """ returns tf.Dataset from generator"""
        gen, args, spec, _ = self.__get_gen_args_spec_dataconfig(mode)
        return tf.data.Dataset.from_generator(gen, args=args, output_signature=spec)

    def data_from_tfrecord(self, mode) -> tf.data.Dataset:
        """ returns tf.Dataset from tfrecord files (only preprocessed) """
        import pickle
        _, _, spec, data_config = self.__get_gen_args_spec_dataconfig(mode)

        # record_file = os.path.join(self.data_config.preprocessed_folder, "preprocessed.tfrecord")
        with open(os.path.join(data_config.preprocessed_folder, 'tfrecord', "dtypes_preprocessed.bin"), 'rb') as F:
            dtype_dict = pickle.load(F)

        output_names = [x.name for x in spec]

        feature_description = {name: tf.io.FixedLenFeature([], tf.string) for name in output_names}

        def get_data_as_tuple(data):
            return tuple((tf.io.parse_tensor(data[name], dtype_dict[name]) for name in output_names))

        def parse_tfrecord(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)

        # SHARDED tfrecord
        files = tf.io.matching_files(
            os.path.join(data_config.preprocessed_folder, 'tfrecord', "preprocessed_*.tfrecord"))
        files = tf.random.shuffle(files)
        shards = tf.data.Dataset.from_tensor_slices(files)
        tf_ds = shards.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'),
                                  num_parallel_calls=tf.data.AUTOTUNE)

        tf_ds = tf_ds.shuffle(buffer_size=1000)

        return tf_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE).map(get_data_as_tuple, num_parallel_calls=tf.data.AUTOTUNE)

    def __get_gen_args_spec_dataconfig(self, mode):
        data_name = self.params.dataset_params.data_name
        cls_type = self.params.dataset_params.cls_type
        use_preprocessed = self.params.dataset_params.use_preprocessed
        size_all = self.params.dataset_params.size_all
        train_size = self.params.dataset_params.train_size
        augment_per_image = self.params.dataset_params.augment_per_image

        if self.params.network == Networks.darknet:

            strides = None # todo might be useless
            anchors = None
            crop_image = False  # does not make sense for yolo
            shuffle = True

            args = (
                mode, data_name, cls_type, use_preprocessed, strides, anchors, size_all, train_size, crop_image,
                shuffle,
                augment_per_image)

            if self.params.dataset_params.dataset == Datasets.blender:
                from lib.data.blender.blender_yolo import BlenderYolo
                from lib.data.blender.blender_settings import BlenderSettings
                gen = BlenderYolo
                gen_data_config = BlenderSettings

            elif self.params.dataset_params.dataset == Datasets.linemod:
                from lib.data.linemod.linemod_yolo import LineModYolo
                from lib.data.linemod.linemod_settings import LineModSettings
                gen = LineModYolo
                gen_data_config = LineModSettings
            elif self.params.dataset_params.dataset == Datasets.unity_grocieries_real:
                from lib.data.unity_groceries_real.unity_groceries_real_yolo import UGRealYolo
                from lib.data.unity_groceries_real.unity_groceries_real_settings import UGRealSettings
                gen = UGRealYolo
                gen_data_config = UGRealSettings
            else:
                raise AssertionError(f"Unknown dataset for {self.params.network}: ", self.params.dataset_params.dataset)

            data_config = gen_data_config(data_name, cls_type, use_preprocessed, crop_image,
                                          size_all, train_size, augment_per_image=augment_per_image)
            from lib.data.dataset_params import yolo_tensor_spec
            spec = yolo_tensor_spec(data_config.n_classes, data_config.yolo_rgb_shape, mode)

        elif self.params.network == Networks.pvn3d:
            from lib.data.dataset_params import pvn3d_tensor_spec
            crop_image = self.params.dataset_params.crop_image
            args = (mode, data_name, cls_type, use_preprocessed, size_all, train_size,
                    crop_image, augment_per_image)

            if self.params.dataset_params.dataset == Datasets.blender:
                from lib.data.blender.blender_pvn import BlenderPvn3d
                from lib.data.blender.blender_settings import BlenderSettings
                gen = BlenderPvn3d
                gen_data_config = BlenderSettings
            elif self.params.dataset_params.dataset == Datasets.linemod:
                from lib.data.linemod.linemod_pvn import LineModPvn3d
                from lib.data.linemod.linemod_settings import LineModSettings
                gen = LineModPvn3d
                gen_data_config = LineModSettings
            else:
                raise AssertionError(f"Unknown dataset for {self.params.network}: ", self.params.dataset_params.dataset)

            data_config = gen_data_config(data_name, cls_type, use_preprocessed, crop_image,
                                          size_all, train_size, augment_per_image=augment_per_image)
            spec = pvn3d_tensor_spec(cls_type, data_config.rgb_input_shape, 12288, mode)

        elif self.params.network is None:
            crop_image = False
            args = (mode, data_name, cls_type, use_preprocessed, size_all, train_size,
                    crop_image, True, augment_per_image)
            if self.params.dataset_params.dataset == Datasets.blender:
                from lib.data.blender.blender import Blender
                from lib.data.blender.blender_settings import BlenderSettings
                gen = Blender
                gen_data_config = BlenderSettings

            elif self.params.dataset_params.dataset == Datasets.linemod:
                from lib.data.linemod.linemod import LineMod
                from lib.data.linemod.linemod_settings import LineModSettings
                gen = LineMod
                gen_data_config = LineModSettings

            elif self.params.dataset_params.dataset == Datasets.unity_grocieries_real:
                from lib.data.unity_groceries_real.unity_groceries_real import UGReal
                from lib.data.unity_groceries_real.unity_groceries_real_settings import UGRealSettings
                gen = UGReal
                gen_data_config = UGRealSettings

            else:
                raise AssertionError("Unknown dataset: ", self.params.dataset_params.dataset)


        else:
            raise AssertionError('Unknown network for dataset: ', self.params.network)

        data_config = gen_data_config(data_name, cls_type, use_preprocessed, crop_image,
                                      size_all, train_size, augment_per_image=augment_per_image)
        return gen, args, spec, data_config
