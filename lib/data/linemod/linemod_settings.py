import os
import numpy as np
from lib.data.dataset_settings import DatasetSettings


class LineModSettings(DatasetSettings):
    def __init__(self, data_name, cls_type='duck', use_preprocessed=False, crop_image=False, size_all=0, train_size=0, augment_per_image=0):
        super().__init__(data_name, cls_type, use_preprocessed, size_all, train_size, augment_per_image=augment_per_image)

        self.camera_scale = 1000
        self.ori_rgb_shape = (480, 640, 3)
        self.rgb_input_shape = (80, 80, 3) if crop_image else (480, 640, 3) # inpust shape of resnet
        self.n_classes = 1 + 9 if cls_type == "all" else 1 + 1
        self.n_objects = 1 + 9 if cls_type == "all" else 1 + 1
        self.cls_lst = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.sym_cls_ids = [10, 11]

        #self.n_sample_points = 8192 + 4096 if not crop_image else 1024
        self.n_key_points = 8
        self.n_ctr_points = 1
        self.n_min_points = 400
        self.dim_pcld_features = 6
        self.dim_pcld_xyz = 3
        self.online_rgb_aug = False

        self.obj_dict = {
            'ape': 1,
            'benchvise': 2,
            'cam': 4,
            'can': 5,
            'cat': 6,
            'driller': 8,
            'duck': 9,
            'eggbox': 10,
            'glue': 11,
            'holepuncher': 12,
            'iron': 13,
            'lamp': 14,
            'phone': 15,
            'all': 16
        }

        self.mask_ids = {
            'ape': 21,
            'benchvise': 43,
            'can': 106,
            'cat': 128,
            'driller': 170,
            'duck': 191,
            'eggbox': 213,
            'glue': 234,
            'holepuncher': 255,
        }

        self.mask_value_array = np.array([value for _, value in self.mask_ids.items()])
        self.mask_name_array = np.array([key for key, _ in self.mask_ids.items()])

        self.mask_binary_array = np.array([0, 1])

        self.id2obj_dict = dict(zip(self.obj_dict.values(), self.obj_dict.keys()))

        self.root = os.path.abspath(os.path.join(self.exp_dir, "../../dataset/linemod"))

        self.data_root = os.path.join(self.root, "linemod", "{}".format(self.data_name))

        self.cls_root = os.path.join(self.data_root, f"{self.obj_dict[self.cls_type]:02}")

        self.kps_dir = os.path.join(self.root, "lm_obj_kpts")

        self.mesh_dir = os.path.join(self.root, "lm_obj_mesh")
        
        self.mesh_paths = {self.id2obj_dict[obj_id]:os.path.join(self.mesh_dir, "obj_{:02}.ply".format(obj_id)) for obj_id in self.cls_lst}
        self.mesh_scale = 0.001

        self.intrinsic_matrix = np.array([[572.4114, 0., 325.2611],
                                          [0., 573.57043, 242.04899],
                                          [0., 0., 1.]]).astype(np.float32)

        self.preprocessed_folder = os.path.join(self.data_root, "{:02}/preprocessed".format(self.obj_dict[self.cls_type]))

        self.test_txt_path = os.path.join(self.data_root, "{:02}/test.txt".format(self.obj_dict[self.cls_type]))

        # ================= yolo_configs ======================
        # we don't resize
        self.yolo_default_rgb_h = 480
        self.yolo_default_rgb_w = 640
        self.yolo_rgb_shape = (480, 640, 3)
