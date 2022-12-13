import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D
from lib.lib.pointnet2_utils.pnet2_layers.layers import Pointnet_SA, Pointnet_FP


class PointNet2Params:
    def __init__(self):
        self.bn = False
        self.is_train = True
        self.keep_prob = 0.5
        self.return_features = True


class PointNetModel(Model):
    def __init__(self, params: PointNet2Params, num_classes):
        super(PointNetModel, self).__init__()

        self.params = params
        self.activation = tf.nn.relu
        self.keep_prob = 0.5
        self.num_classes = num_classes
        self.bn = self.params.bn

        self.kernel_initializer = 'glorot_normal'
        self.kernel_regularizer = None
        self.group_all = False

        self.init_network()

    def init_network(self):
        self.sa_1 = Pointnet_SA(
            npoint=512,
            radius=0.1,
            nsample=32,
            mlp=[32, 32, 64],
            group_all=self.group_all,
            activation=self.activation,
            bn=self.bn
        )

        self.sa_2 = Pointnet_SA(
            npoint=128,
            radius=0.2,
            nsample=32,
            mlp=[64, 64, 128],
            group_all=self.group_all,
            activation=self.activation,
            bn=self.bn
        )

        self.sa_3 = Pointnet_SA(
            npoint=32,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 256],
            group_all=self.group_all,
            activation=self.activation,
            bn=self.bn
        )

        self.sa_4 = Pointnet_SA(
            npoint=8,
            radius=0.8,
            nsample=32,
            mlp=[256, 256, 512],
            group_all=self.group_all,
            activation=self.activation,
            bn=self.bn
        )

        self.fp_1 = Pointnet_FP(
            mlp=[256, 256],
            activation=self.activation,
            bn=self.bn
        )

        self.fp_2 = Pointnet_FP(
            mlp=[256, 256],
            activation=self.activation,
            bn=self.bn
        )

        self.fp_3 = Pointnet_FP(
            mlp=[256, 128],
            activation=self.activation,
            bn=self.bn
        )

        self.fp_4 = Pointnet_FP(
            mlp=[128, 128, 128],
            activation=self.activation,
            bn=self.bn
        )

        self.conv1d = Conv1D(filters=self.num_classes, kernel_size=1, activation=None)

    def call(self, inputs, training=None, mask=None):
        l0_xyz = inputs[0]
        l0_points = inputs[1]

        l1_xyz, l1_points = self.sa_1(l0_xyz, l0_points, training=training)
        l2_xyz, l2_points = self.sa_2(l1_xyz, l1_points, training=training)
        l3_xyz, l3_points = self.sa_3(l2_xyz, l2_points, training=training)
        l4_xyz, l4_points = self.sa_4(l3_xyz, l3_points, training=training)

        l3_points = self.fp_1(l3_xyz, l4_xyz, l3_points, l4_points, training=training)
        l2_points = self.fp_2(l2_xyz, l3_xyz, l2_points, l3_points, training=training)
        l1_points = self.fp_3(l1_xyz, l2_xyz, l1_points, l2_points, training=training)
        l0_points = self.fp_4(l0_xyz, l1_xyz, l0_points, l1_points, training=training)

        if self.params.return_features:
            return l0_points
        else:
            seg_features = self.conv1d(l0_points)
            final_seg = tf.nn.softmax(seg_features)
            return final_seg
