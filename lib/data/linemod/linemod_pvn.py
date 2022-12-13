from lib.data.network_mixins.pvn_mixin import PvnMixin
from lib.data.linemod.linemod import LineMod


class LineModPvn3d(PvnMixin, LineMod):
    def __init__(self, mode, data_name, cls_type, use_preprocessed, size_all, train_size, crop_image, augment_per_image=0, use_pvn_kp=False,
                 shuffle=True):
        super().__init__(use_pvn_kp, mode, data_name, cls_type, use_preprocessed, size_all, train_size, crop_image, augment_per_image=augment_per_image, shuffle=shuffle)