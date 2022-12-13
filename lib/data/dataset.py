from builtins import Exception
import os
import random
import numpy as np
from lib.data.dataset_settings import DatasetSettings
from lib.utils import CachedProperty
from lib.data.utils import convert2pascal
from abc import abstractmethod


class NoMaskError(Exception):
    pass


class NoDepthError(Exception):
    pass


class NoRtError(Exception):
    pass


class Dataset:
    data_config: DatasetSettings

    def __init__(self, data_config: DatasetSettings, mode, data_name, cls_type,
                 crop_image=False, shuffle=True, *args, **kwargs):

        super(Dataset, self).__init__(*args, **kwargs)

        self.data_config = data_config
        self.dataset_name = data_name
        self.mode = mode.decode("utf-8") if isinstance(mode, bytes) else mode
        self.cls_type = self.data_config.cls_type
        self.crop_image = crop_image
        self.counter = 0
        self.shuffle = shuffle
        self.demo_size = 10

        self.if_augment = self.data_config.augment_per_image > 0

        self.cls_id = self.data_config.obj_dict[self.cls_type]
        self.cls_root = self.data_config.cls_root
        self.obj_cls_id = 1 if cls_type != "all" else self.cls_id

        # using preprocessed when preprocessing doesnt make sense
        if self.mode == "preprocess":
            self.data_config.use_preprocessed = False

    @abstractmethod
    def get_rgb(self, index) -> np.ndarray:
        """ get image in rgb as ndarray"""
        pass

    @abstractmethod
    def get_gt_bbox(self, index) -> np.ndarray:
        """ get bbox for cls for index """
        pass

    @abstractmethod
    def get_depth(self, index) -> np.ndarray:
        """ get depth image in rgb as ndarray, scaled to m"""
        pass

    @abstractmethod
    def get_mask(self, index) -> np.ndarray:
        """ get mask. If cls_type != 'all' return mask for cls_type with {0, 255}
                      If cls_type == 'all' return mask with mask_ids
        """
        pass

    @abstractmethod
    def get_num_imgs(self) -> int:
        """ get number of raw images """
        pass

    @abstractmethod
    def get_RT_list(self, index):
        """ get list of Rt (tuples of RT matrix and cls_id)"""
        pass

    @abstractmethod
    def get(self, index):
        """ get training data as tuple """
        pass

    @abstractmethod
    def get_dict(self, index):
        """ get dict with network relevant info"""
        pass

    def get_item(self):
        """ retrieve item for iterator"""
        index = self.index_lst[self.counter]

        # if not self.data_config.use_preprocessed and self.if_augment:
        # translate augmented index into original image index
        # index = index // self.data_config.augment_per_image

        return self.get(index)

    @CachedProperty
    def index_lst(self):
        if not self.data_config.use_preprocessed:
            if self.mode == 'train':
                index_lst = range(0, self.data_config.train_size)
            elif self.mode == 'val':
                index_lst = range(self.data_config.train_size, self.data_config.size_all)
            elif self.mode == 'preprocess':
                index_lst = range(0, self.data_config.size_all)
        else:
            # read n_datapoints from preprocessed data
            if self.mode == 'train':
                import json
                with open(os.path.join(self.data_config.preprocessed_folder, 'tfrecord', "meta.json"), 'rb') as F:
                    meta = json.load(F)
                index_lst = range(0, meta['n_datapoints'])
            elif self.mode == 'val':
                n_datapoints = len(os.listdir(os.path.join(self.data_config.preprocessed_folder, 'rgb')))
                index_lst = range(0, n_datapoints)
            elif self.mode == 'preprocess':
                raise AssertionError('Reading a preprocessed dataset to preprocess a dataset does not make sense.')

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

    def __len__(self):
        return len(self.index_lst)
