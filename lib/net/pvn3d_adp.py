from tensorflow.keras import Model
from lib.net.resnet import ResNetParams, ResNet
from lib.net.pspnet import PspNetParams, PspNet
from lib.net.pointnet2_tf import PointNet2TfModel, PointNet2Params
from lib.net.pointnet2 import PointNetModel
from lib.net.densefusion import DenseFusionNet, DenseFusionNetParams
from lib.net.mlp import MlpNets, MlpNetsParams
from lib.net.utils import match_choose_adp
import tensorflow as tf


class Pvn3dNetParams:
    def __init__(self):
        self.resnet_params = ResNetParams()
        self.psp_params = PspNetParams()
        self.point_net2_params = PointNet2Params()
        self.dense_fusion_params = DenseFusionNetParams()
        self.mlp_params = MlpNetsParams()


class Pvn3dNet(Model):
    params: Pvn3dNetParams
    
    def __init__(self, params: Pvn3dNetParams, rgb_input_shape,
                 num_kpts, num_cls, num_cpts, dim_xyz):

        super(Pvn3dNet, self).__init__()
        self.params = params

        self.num_pts = num_pts = params.point_net2_params.n_sample_points

        self.resnet_input_shape = rgb_input_shape
        self.resnet_net = ResNet(self.params.resnet_params, rgb_input_shape)
        self.resnet_model = self.resnet_net.build_resnet_model()

        self.psp_net = PspNet(self.params.psp_params)
        self.psp_model = self.psp_net.build_psp_model(list(self.resnet_model.output_shape)[1:])

        if self.params.point_net2_params.use_tf_interpolation:
            self.pointnet2_model = PointNet2TfModel(self.params.point_net2_params, num_classes=num_cls)
        else:
            self.pointnet2_model = PointNetModel(self.params.point_net2_params, num_classes=num_cls)

        self.dense_fusion_net = DenseFusionNet(self.params.dense_fusion_params)
        self.dense_fusion_model = self.dense_fusion_net.build_dense_fusion_model(
            rgb_emb_shape=(num_pts, self.params.dense_fusion_params.num_embeddings),
            pcl_emb_shape=(num_pts, self.params.dense_fusion_params.num_embeddings))

        self.mlp_net = MlpNets(self.params.mlp_params,
                               num_pts=num_pts,
                               num_kpts=num_kpts,
                               num_cls=num_cls,
                               num_cpts=num_cpts,
                               channel_xyz=dim_xyz)

        self.num_rgbd_feats = list(self.dense_fusion_model.output_shape)[-1]
        self.mlp_model = self.mlp_net.build_mlp_model(rgbd_features_shape=(num_pts, self.num_rgbd_feats))

    def call(self, inputs, training=None, mask=None):

        pcld, sampled_index, rgb, crop_factor = inputs # resized for resnet -> crop factor

        feats = self.resnet_model(rgb, training=training)
        pcld_emb = self.pointnet2_model(pcld, training=training)
        rgb_features = self.psp_model(feats, training=training)

        rgb_emb = match_choose_adp(rgb_features, sampled_index, crop_factor, self.resnet_input_shape)

        feats_fused = self.dense_fusion_model([rgb_emb, pcld_emb], training=training)
        kp, sm, cp = self.mlp_model(feats_fused, training=training)

        return kp, sm, cp


# todo adapt the network.forward_pass() to static method, such that it can be used for evaluation directly
@tf.function
def forward_pass(input_data, model, training=False):
    """
    params:
            rgb [bs, H, W, 3]
            pcld_xyz [bs, npts, 3]
            pcld_feats [bs, npts, 6]
            sampled_index [bs, npts]
    output:
        kpts_pre [bs, npts, n_kpts, 3]
        cpts_pre [bs, npts, n_cpts, 3]
        segs_pre [bs, npts, n_cls]
    """
    rgb, pcld_xyz, pcld_feats, sampled_index, crop_factor = input_data

    pcld = [pcld_xyz, pcld_feats]
    inputs = [pcld, tf.cast(sampled_index, tf.int32), rgb, tf.cast(crop_factor, tf.int32)]

    kp_pre_ofst, seg_pre, cp_pre_ofst = model(inputs, training=training)

    return kp_pre_ofst, seg_pre, cp_pre_ofst
