import os
import numpy as np
import json
from lib.data.dataset_settings import DatasetSettings


class UGRealSettings(DatasetSettings):
    def __init__(self, data_name, cls_type=None, use_preprocessed=False, crop_image=False, size_all=0, train_size=0,
                 augment_per_image=0):
        super().__init__(data_name, cls_type, use_preprocessed, size_all, train_size,
                         augment_per_image=augment_per_image)

        # ================ original_data_configs =============
        self.ori_rgb_shape = (3456, 5184, 3)

        # ================ pvn3d_configs ================
        self.camera_scale = 1
        self.rgb_input_shape = (240, 240, 3) if crop_image else (480, 640, 3)  # inpust shape of resnet

        # self.n_sample_points = 8192 + 4096 if not crop_image else 1024
        self.n_key_points = 8
        self.n_ctr_points = 1
        self.n_min_points = 400
        self.dim_pcld_features = 6
        self.dim_pcld_xyz = 3
        self.online_rgb_aug = False

        self.root = os.path.abspath(os.path.join(self.exp_dir, "../../dataset/UnityGroceriesReal"))
        self.data_root = os.path.join(self.root, "UnityGroceriesReal")

        with open(os.path.join(self.data_root, 'annotation_definitions.json')) as f:
            anno_def = json.load(f)

        definitions = anno_def['annotation_definitions'][0]['spec']
        self.obj_dict = {x['label_name']: x['label_id'] for x in definitions if len(x['label_name'])>0}
        self.obj_dict.update({'all': 1+max(self.obj_dict.values())})
        self.sym_cls_ids = []

        self.cls_lst = [val for key, val in self.obj_dict.items() if key != 'all']

        self.n_classes = len(self.obj_dict) if cls_type == "all" else 1 + 1
        self.n_objects = len(self.obj_dict) if cls_type == "all" else 1 + 1

        self.mask_ids = {key:val for key, val in self.obj_dict.items() if key != 'all'} # for blender!

        self.mask_value_array = np.array([value for _, value in self.mask_ids.items()])
        self.mask_name_array = np.array([key for key, _ in self.mask_ids.items()])

        self.mask_binary_array = np.array([0, 1])

        self.id2obj_dict = dict(zip(self.obj_dict.values(), self.obj_dict.keys()))

        self.cls_root = self.data_root

        self.intrinsic_matrix = None

        self.kps_dir = None
        self.mesh_dir = os.path.join(self.root, 'models')
        
        self.mesh_paths = {}
        self.mesh_scale = 1.0

        self.preprocessed_folder = os.path.join(self.data_root, 'preprocessed')

        # ================= yolo_configs ======================
        self.yolo_default_rgb_h = 416
        self.yolo_default_rgb_w = 416
        self.yolo_rgb_shape = (416, 416, 3)
