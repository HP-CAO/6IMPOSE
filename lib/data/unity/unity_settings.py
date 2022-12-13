import os
import numpy as np


class UnitySettings:
    def __init__(self, data_name, cls_type='', use_preprocessed=False, crop_image=False, size_all=0, train_size=0):
        # ================ original_data_configs =============

        self.ori_rgb_shape = (3840, 2160, 3)
        self.data_name = data_name.decode("utf-8") if isinstance(data_name, bytes) else data_name

        # ================ pvn3d_configs ================
        self.camera_scale = 1

        self.rgb_input_shape = (240, 240, 3) if crop_image else (480, 640, 3) # inpust shape of resnet

        self.n_classes = 1 + 9 if cls_type == "all" else 1 + 1
        self.n_objects = 1 + 9 if cls_type == "all" else 1 + 1
        self.cls_lst = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.sym_cls_ids = [10, 11]
        self.cls_type = cls_type
        self.n_sample_points = 8192 + 4096 if not crop_image else 1024 * 3
        self.n_key_points = 8
        self.n_ctr_points = 1
        self.n_min_points = 400
        self.dim_pcld_features = 6
        self.dim_pcld_xyz = 3
        self.use_preprocessed = use_preprocessed
        self.size_all = size_all
        self.train_size = train_size
        self.exp_dir = os.path.dirname(__file__)
        self.exp_name = os.path.basename(self.exp_dir)
        self.mesh_factor = 1.0
        self.online_rgb_aug = False

        self.obj_dict = {
            'duck': 1,
            'banana': 2,
            'can': 3,
            'jug': 4,
            'all': 10
        }

        self.mask_ids = {
            'duck': 1,
            'banana': 1,
            'can': 1,
            'jug': 1,
            'all': 1
        }

        self.mask_value_array = np.array([value for _, value in self.mask_ids.items()])
        self.mask_name_array = np.array([key for key, _ in self.mask_ids.items()])

        self.mask_binary_array = np.array([0, 1])

        self.id2obj_dict = dict(zip(self.obj_dict.values(), self.obj_dict.keys()))

        self.root = os.path.abspath(
            os.path.join(self.exp_dir, "../../../dataset/unity/unity/{}".format(self.data_name)))
        self.kps_dir = os.path.abspath(os.path.join(self.exp_dir, '../../../dataset/unity/unity_obj_kpts'))
        self.mesh_dir = os.path.abspath(os.path.join(self.exp_dir, '../../../dataset/unity/unity_obj_mesh'))

        self.intrinsic_matrix = np.array([[1870.6150197846725, 0, 1920.0],
                                         [0, 1870.6148721743875, 1080.0],
                                         [0, 0, 1]])

        self.preprocessed_folder = os.path.join(self.root, "{:02}/preprocessed".format(self.obj_dict[self.cls_type]))

        # ================= yolo_configs ======================
        self.yolo_default_rgb_h = 416
        self.yolo_default_rgb_w = 416
        self.yolo_rgb_shape = (416, 416, 3)
