import functools
from segmentation_models.backbones.backbones_factory import Backbones
from tensorflow import keras
from tensorflow.keras.models import Model

_KERAS_BACKEND = keras.backend
_KERAS_LAYERS = keras.layers
_KERAS_MODELS = keras.models
_KERAS_UTILS = keras.utils


class BackboneParams:
    def __init__(self):
        self.backbone_type = 'resnet18'
        self.downsample_factor = 6
        self.backbone_weights = 'imagenet'
        self.freeze_model = True
        self.include_top = False


def inject_global_submodules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = _KERAS_BACKEND
        kwargs['layers'] = _KERAS_LAYERS
        kwargs['models'] = _KERAS_MODELS
        kwargs['utils'] = _KERAS_UTILS
        return func(*args, **kwargs)

    return wrapper


def get_backbone_names():
    return Backbones.models_names()


def get_backbone_model(name='seresnet101', input_shape=(480, 640, 3), weights='imagenet',
                       include_top=False, downsample_factor=16, freeze_model=False):

    if name not in get_backbone_names():
        print(get_backbone_names())
        raise NameError('model is not in the model_list')

    backbone = inject_global_submodules(Backbones.get_backbone)(name=name,
                                                                input_shape=input_shape,
                                                                weights=weights,
                                                                include_top=include_top)

    if freeze_model:
        for layer in backbone.layers:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

    feature_layers = Backbones.get_feature_layers(name, n=3)

    if downsample_factor == 16:
        layer_idx = feature_layers[0]
    elif downsample_factor == 8:
        layer_idx = feature_layers[1]
    elif downsample_factor == 4:
        layer_idx = feature_layers[2]
    elif downsample_factor == 32:  # means the original structure
        return backbone
    else:
        raise ValueError('Unsupported factor - `{}`, Use 4, 8, 16, 32.'.format(downsample_factor))

    input_ = backbone.input

    output = (backbone.get_layer(name=layer_idx).output if isinstance(layer_idx, str)
         else backbone.get_layer(index=layer_idx).output)

    model = Model(input_, output, name=name + '_backbone')

    return model


if __name__ == '__main__':
    # short example here
    BackboneModel = get_backbone_model(freeze_model=True)
    BackboneModel.summary()
