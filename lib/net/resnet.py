from tensorflow.keras import Input, Model
from lib.net.backbone import get_backbone_model


class ResNetParams:
    def __init__(self):
        self.backbone_type = 'resnet18'
        self.down_sample_factor = 8
        self.backbone_weights = 'imagenet'
        self.interpolation_method = 'nearest'
        self.freeze_model = True
        self.include_top = False


class ResNet:
    def __init__(self, params: ResNetParams, input_shape):
        super(ResNet, self).__init__()
        self.params = params
        self.input_shape = input_shape
        self.crop_factor = 1

    def build_resnet_model(self):
        input_layer = Input(shape=self.input_shape, name='resnet_rgb_input')
        resnet_model = get_backbone_model(name=self.params.backbone_type,
                                          input_shape=self.input_shape,
                                          downsample_factor=self.params.down_sample_factor,
                                          weights=self.params.backbone_weights,
                                          freeze_model=self.params.freeze_model,
                                          include_top=self.params.include_top)

        output_features = resnet_model(input_layer)
        model = Model(inputs=input_layer, outputs=output_features)
        return model
