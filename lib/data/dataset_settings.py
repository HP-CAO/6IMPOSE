from abc import ABC
import numpy as np
import os
from typing import Dict, Tuple, List


class DatasetSettings(ABC):
    preprocessed_folder: str
    intrinsic_matrix: np.ndarray
    size_all: int
    train_size: int
    data_name: str
    use_preprocessed: bool
    obj_dict: Dict[str, int]
    augment_per_image: int
    cls_root: str
    mesh_paths: Dict[str, str]
    mesh_scale: float  # from mesh to m
    ori_rgb_shape: Tuple
    n_classes: int
    n_objects: int
    cls_lst: List[int]

    def __init__(self, data_name, cls_type, use_preprocessed, size_all, train_size, augment_per_image=0):
        self.data_name = data_name.decode("utf-8") if isinstance(data_name, bytes) else data_name
        self.cls_type = cls_type.decode("utf-8") if isinstance(cls_type, bytes) else cls_type
        self.use_preprocessed = use_preprocessed

        self.exp_dir = os.path.dirname(__file__)

        self.size_all = size_all
        self.train_size = train_size

        self.augment_per_image = augment_per_image
