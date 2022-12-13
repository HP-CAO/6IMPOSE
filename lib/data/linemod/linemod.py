import os
import yaml
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from lib.data.linemod.linemod_settings import LineModSettings
from lib.utils import CachedProperty
from lib.data.utils import convert2pascal

from lib.data.dataset import Dataset

class LineMod(Dataset):
    def __init__(self, mode, data_name, cls_type, use_preprocessed, size_all, train_size,
                 crop_image=False, shuffle=True, augment_per_image=0, *args, **kwargs):

        data_config = LineModSettings(data_name, cls_type, use_preprocessed, crop_image, size_all, train_size, augment_per_image=augment_per_image)
        
        super().__init__(data_config, mode, data_name, cls_type, crop_image, shuffle=shuffle, *args, **kwargs)


    def get_rgb(self, index):
        cls_root = self.cls_root
        rgb_path = os.path.join(cls_root, "rgb", f"{index:04}.png")

        with Image.open(rgb_path) as rgb:
            rgb = np.array(rgb).astype(np.uint8)

        return rgb

    def get_num_imgs(self) -> int:
        cls_root = self.cls_root
        rgb_path = os.path.join(cls_root, "rgb")
        return len(os.listdir(rgb_path))
        


    def get_gt_bbox(self, index):
        """ list of bboxes: [x1, y1, x2, y2, cls_id] """
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


    def get_depth(self, index):
        with Image.open(os.path.join(self.cls_root, "depth", f"{index:04}.png")) as depth:
            dpt = np.array(depth).astype(np.float32)
            return dpt / self.data_config.camera_scale

    def get_mask(self, index):
        with Image.open(os.path.join(self.cls_root, "mask", f"{index:04}.png")) as mask:
            mask = np.array(mask)
        return mask

    def get_RT_list(self, index):
        """
        return a list of tuples of RT matrix and cls_id [(RT_0, cls_id_0), (RT_1,, cls_id_1) ..., (RT_N, cls_id_N)]
        """
        meta_list = []
        RT_list = []
        meta = self.meta_lst[index]

        if self.cls_id == 2:
            for i in range(0, len(meta)):
                if meta[i]['obj_id'] == 2:
                    meta_list.append(meta[i])
                    break
        elif self.cls_id == 16: # all
            for i in range(0, len(meta)):
                meta_list.append(meta[i])
        else:
            meta_list.append(meta[0])

        for mt in meta_list:
            R = np.resize(np.array(mt['cam_R_m2c']), (3, 3))
            T = np.array(mt['cam_t_m2c']) / 1000.0
            cls_id = mt['obj_id']
            RT = np.concatenate((R, T[:, None]), axis=1)
            RT_list.append((RT, cls_id))

        return RT_list


    @CachedProperty
    def meta_lst(self):
        with open(os.path.join(self.cls_root, 'gt.yml'), "r") as meta_file:
            return yaml.load(meta_file, Loader=yaml.FullLoader)
            
    #@CachedProperty
    #def size_all(self):
    #    size = len(self.meta_lst)
    #    if self.data_config.use_preprocessed:
    #        size = len(os.listdir(self.data_config.preprocessed_folder))
    #    return size