from lib.data.linemod.linemod import LineMod
from lib.data.network_mixins.yolo_mixin import YoloMixin



class LineModYolo(YoloMixin, LineMod):
    def __init__(self, mode, data_name, cls_type, use_preprocessed, strides, anchors, size_all,
                 train_size, crop_image, shuffle=True, augment_per_image=0, *args, **kwargs):
                 
        super().__init__(strides, anchors, mode, data_name, cls_type, use_preprocessed, size_all, train_size,
                                    crop_image, shuffle=shuffle, augment_per_image=augment_per_image)