import os
import numpy as np
from PIL import Image
from lib.data.blender.blender_settings import BlenderSettings
from lib.data.dataset import Dataset
import cv2
import json
from scipy.spatial.transform import Rotation as R
from lib.data.utils import  get_bbox_from_mask


class Blender(Dataset):
    def __init__(self, mode, data_name, cls_type, use_preprocessed, size_all, train_size,
                 crop_image=False, shuffle=True, augment_per_image=0, *args, **kwargs):

        data_config = BlenderSettings(data_name, cls_type, use_preprocessed, crop_image, size_all, train_size, augment_per_image=augment_per_image)

        super().__init__(data_config, mode, data_name, cls_type, crop_image, shuffle=shuffle, *args, **kwargs)

        if self.data_config.cls_type == 'all' and self.data_config.data_name.endswith('_linemod'):
            self.current_cls_root = 0

    def get_cls_root(self, index):
        if self.data_config.cls_type == 'all' and self.data_config.data_name.endswith('_linemod'):
            # choose an cls dataset in RR fashion
            cls_id = self.data_config.cls_lst[self.current_cls_root % len(self.data_config.cls_lst)]
            cls_root = self.cls_root[cls_id]
            self.current_cls_root += 1
        else:
            cls_root = self.cls_root
        return cls_root

    def get_num_imgs(self) -> int:
        cls_root = self.get_cls_root(0)
        rgb_path = os.path.join(cls_root, "rgb")
        return len(os.listdir(rgb_path))


    def get_rgb(self, index):
        cls_root = self.get_cls_root(index)
        rgb_path = os.path.join(cls_root, "rgb", f"rgb_{index:04}.png")

        try:
            with Image.open(rgb_path) as rgb:
                rgb = np.array(rgb).astype(np.uint8)
        except OSError:
            print("\nCOUD NOT OPEN IMAGE: ", rgb_path)
            rgb = None
            
        return rgb

    def get_mask(self, index):
        mask_path = os.path.join(self.get_cls_root(index), "mask", f"segmentation_{index:04}.exr")

        mask = cv2.imread(mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        mask = mask[:, :, 2].astype(np.uint8)

        if self.cls_type != 'all':
            # for blender datasets: mask id == cls_id
            mask_id = self.data_config.obj_dict[self.data_config.cls_type]
            mask[mask != mask_id] = 0
            mask[mask == mask_id] = 255

        return mask.astype(np.uint8)
    

    def get_depth(self, index):
        depth_path = os.path.join(self.get_cls_root(index), "depth", f"depth_{index:04}.exr")

        dpt = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        dpt = dpt[:, :, 2]
        dpt_mask = dpt < 5  # in meters, we filter out the background( > 5m)
        dpt = dpt * dpt_mask

        return dpt

    def get_RT_list(self, index):
        """ return a list of tuples of RT matrix and cls_id [(RT_0, cls_id_0), (RT_1,, cls_id_1) ..., (RT_N, cls_id_N)] """
        with open(os.path.join(self.get_cls_root(index), "gt", f"gt_{index:05}.json")) as f:
            shot = json.load(f)

        cam_quat = shot['cam_rotation']
        cam_rot = R.from_quat([*cam_quat[1:], cam_quat[0]])
        cam_pos = np.array(shot['cam_location'])
        cam_Rt = np.eye(4)
        cam_Rt[:3, :3] = cam_rot.as_matrix().T
        cam_Rt[:3, 3] = -cam_rot.as_matrix() @ cam_pos

        objs = shot['objs']

        RT_list = []

        if self.cls_type == 'all':
            for obj in objs:
                cls_type = obj['name']
                cls_id = self.data_config.obj_dict[cls_type]
                pos = np.array(obj['pos'])
                quat = obj['rotation']
                rot = R.from_quat([*quat[1:], quat[0]])
                Rt = np.eye(4)
                Rt[:3, :3] = cam_rot.as_matrix().T @ rot.as_matrix()
                Rt[:3, 3] = cam_rot.as_matrix().T @ (pos - cam_pos)
                Rt = Rt[:3, :]  # -> (3, 4)
                RT_list.append((Rt, cls_id))
    
        else:
            for obj in objs:  # here we only consider the single obj
                if (obj['name'] == self.cls_type):
                    cls_type = obj['name']
                    cls_id = self.data_config.obj_dict[cls_type]
                    pos = np.array(obj['pos'])
                    quat = obj['rotation']
                    rot = R.from_quat([*quat[1:], quat[0]])
                    Rt = np.eye(4)
                    Rt[:3, :3] = cam_rot.as_matrix().T @ rot.as_matrix()
                    Rt[:3, 3] = cam_rot.as_matrix().T @ (pos - cam_pos)
                    Rt = Rt[:3, :]  # -> (3, 4)
                    RT_list.append((Rt, cls_id))
        return RT_list
    
    def get_gt_bbox(self, index) -> np.ndarray:
        bboxes = []
        mask = self.get_mask(index)
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
            bbox = list(bbox)
            bbox.append(self.cls_id)
            bboxes.append(bbox)

        return np.array(bboxes)
