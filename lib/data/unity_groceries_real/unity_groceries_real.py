import os
import numpy as np
from PIL import Image
from lib.data.unity_groceries_real.unity_groceries_real_settings import UGRealSettings
from lib.data.dataset import Dataset, NoDepthError, NoRtError, NoMaskError
import cv2
import json
from scipy.spatial.transform import Rotation as R
from lib.data.utils import  convert2pascal, get_bbox_from_mask
from lib.utils import CachedProperty



class UGReal(Dataset):
    def __init__(self, mode, data_name, cls_type, use_preprocessed, size_all, train_size,
                 crop_image=False, shuffle=True, augment_per_image=0, *args, **kwargs):

        data_config = UGRealSettings(data_name, cls_type, use_preprocessed, crop_image, size_all, train_size, augment_per_image=augment_per_image)

        super().__init__(data_config, mode, data_name, cls_type, crop_image, shuffle=shuffle, *args, **kwargs)

    @CachedProperty
    def meta_lst(self):
        with open(os.path.join(self.cls_root, 'annotations.json'), "r") as meta_file:
            return json.load(meta_file)
    
    def get_num_imgs(self) -> int:
        rgb_path = os.path.join(self.cls_root, "images")
        return len(os.listdir(rgb_path))


    def get_rgb(self, index):
        rgb_path = os.path.join(self.cls_root, "images", self.meta_lst[index]["file_name"])

        with Image.open(rgb_path) as rgb:
            rgb = np.array(rgb).astype(np.float32)
            return rgb

    def get_mask(self, index):
        raise NoMaskError("No mask available for Unity Groceries Real")
    

    def get_depth(self, index):
        raise NoDepthError("No depth available for Unity Groceries Real")
        

    def get_RT_list(self, index):
        raise NoRtError("No RT available for Unity Groceries Real")

    
    def get_gt_bbox(self, index) -> np.ndarray:

        all_bboxes = self.meta_lst[index]["annotations"]

        if self.cls_type != 'all':
            all_bboxes = [x for x in all_bboxes if x["label_id"==self.cls_id]]

        bboxes = np.array([[*convert2pascal(x["bbox"]), x["label_id"]] for x in all_bboxes])

        return bboxes
        
