from lib.data.network_mixins.pvn_mixin import PvnMixin
from lib.utils import CachedProperty
from lib.data.blender.blender import Blender


class BlenderPvn3d(PvnMixin, Blender):
    def __init__(self, mode, data_name, cls_type, use_preprocessed, size_all, train_size, crop_image, augment_per_image=0, use_pvn_kp=False,
                 shuffle=True):
        super().__init__(use_pvn_kp, mode, data_name, cls_type, use_preprocessed, size_all, train_size, crop_image, shuffle=shuffle,
                         augment_per_image=augment_per_image)