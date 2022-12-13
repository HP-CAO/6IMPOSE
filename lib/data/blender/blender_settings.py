import os
import numpy as np
import json
from lib.data.dataset_settings import DatasetSettings


class BlenderSettings(DatasetSettings):
    def __init__(self, data_name, cls_type='duck', use_preprocessed=False, crop_image=False, size_all=0, train_size=0,
                 augment_per_image=0):
        super().__init__(data_name, cls_type, use_preprocessed, size_all, train_size,
                         augment_per_image=augment_per_image)

        # ================ original_data_configs =============
        self.ori_rgb_shape = (1080, 1920, 3)
        if self.data_name.endswith('_linemod'):
            self.ori_rgb_shape = (480, 640, 3)

        # ================ pvn3d_configs ================
        self.camera_scale = 1
        self.rgb_input_shape = (80, 80, 3) if crop_image else (480, 640, 3)  # inpust shape of resnet

        # self.n_sample_points = 8192 + 4096 if not crop_image else 1024
        self.n_key_points = 8
        self.n_ctr_points = 1
        self.n_min_points = 400
        self.dim_pcld_features = 6
        self.dim_pcld_xyz = 3
        self.online_rgb_aug = False

        # If we use blender_linemod overwrite obj_dict with linemod dict
        if self.data_name.endswith('_linemod'):
            from lib.data.linemod.linemod_settings import LineModSettings
            lm_settings = LineModSettings('data')  # arg doesnt matter, only need obj_dict
            self.obj_dict = lm_settings.obj_dict
            self.sym_cls_ids = [10, 11]
        elif self.data_name.endswith('_ugreal'):
            from lib.data.unity_groceries_real.unity_groceries_real_settings import UGRealSettings
            ugreal_settings = UGRealSettings('data')
            self.obj_dict = ugreal_settings.obj_dict
            self.sym_cls_ids = []
        else:
            self.obj_dict = {
                'cpsduck': 1,
                'stapler': 2,
                'all': 100
            }
            self.sym_cls_ids = []

        self.cls_lst = [val for key, val in self.obj_dict.items() if key != 'all']

        self.n_classes = len(self.obj_dict) if cls_type == "all" else 1 + 1
        self.n_objects = len(self.obj_dict) if cls_type == "all" else 1 + 1

        self.mask_ids = {key: val for key, val in self.obj_dict.items() if key != 'all'}  # for blender!

        self.mask_value_array = np.array([value for _, value in self.mask_ids.items()])
        self.mask_name_array = np.array([key for key, _ in self.mask_ids.items()])

        self.mask_binary_array = np.array([0, 1])

        self.id2obj_dict = dict(zip(self.obj_dict.values(), self.obj_dict.keys()))

        self.root = os.path.abspath(os.path.join(self.exp_dir, "../../dataset/blender"))

        self.background_dir = os.path.join(self.root, "backgrounds")

        self.data_root = os.path.join(self.root, "blender", "{}".format(self.data_name))

        if self.cls_type == 'all' and self.data_name.endswith('_linemod'):
            # in this case load all obj datasets to interweave
            self.cls_root = {cls_id: os.path.join(self.data_root, f"{cls_id:02}") for cls_id in self.cls_lst}
            with open(os.path.join(self.cls_root[self.cls_lst[0]], "gt.json")) as f:
                json_dict = json.load(f)
        else:
            self.cls_root = os.path.join(self.data_root, f"{self.obj_dict[self.cls_type]:02}")
            with open(os.path.join(self.cls_root, "gt.json")) as f:
                json_dict = json.load(f)

        self.intrinsic_matrix = np.array(json_dict["camera_matrix"])

        self.kps_dir = os.path.join(self.root, 'bl_obj_kpts')
        self.mesh_dir = os.path.join(self.root, 'bl_obj_mesh')

        self.mesh_paths = {cls_type: os.path.join(self.mesh_dir, "{}.ply".format(cls_type)) for cls_type in
                           self.obj_dict.keys() if cls_type != 'all'}
        self.mesh_scale = 1.0

        # overwrite mesh and kpts dirs also
        if self.data_name.endswith('_linemod'):
            self.kps_dir = os.path.abspath(os.path.join(self.exp_dir, '../../dataset/linemod/lm_obj_kpts'))
            self.mesh_dir = os.path.abspath(os.path.join(self.exp_dir, '../../dataset/linemod/lm_obj_mesh'))

            # self.mesh_path = os.path.join(self.mesh_dir, "obj_{:02}.ply".format(obj_id))

            self.mesh_paths = {cls_type: os.path.join(self.mesh_dir, "obj_{:02}.ply".format(self.obj_dict[cls_type]))
                               for cls_type in self.obj_dict.keys() if cls_type != 'all'}

            self.mesh_scale = 0.001

        elif self.data_name.endswith('_ugreal'):
            self.mesh_paths = {}

            self.mesh_scale = 1.0

        self.preprocessed_folder = os.path.join(self.data_root,
                                                "{:02}/preprocessed".format(self.obj_dict[self.cls_type]))

        # ================= yolo_configs ======================
        # self.yolo_default_rgb_h = 416
        # self.yolo_default_rgb_w = 416
        # self.yolo_rgb_shape = (416, 416, 3)

        self.yolo_default_rgb_h = 480
        self.yolo_default_rgb_w = 640
        self.yolo_rgb_shape = (480, 640, 3)
