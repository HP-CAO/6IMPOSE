import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Layer
from lib.net.pointnet2_utils.pnet2_layers.layers import Pointnet_SA


@tf.function
def farthest_point_sample_and_group(n_points, xyz, n_samples):
    """
    batch farthest point sampling in tensorflow
    :param n_samples: the number of points in the neighbour of the selected point
    :param n_points: the number of points needs to be sampled
    :param xyz: point cloud input bs, N, 3
    :return: the sampled points bs, n_points, 3
             the index of the neighbouring points of selected points bs, n_points, n_samples
    """
    bs, N, c = xyz.shape
    dist = tf.ones(shape=(bs, N)) * np.inf
    sample_mask = tf.zeros(shape=(bs, N))
    sample_id = tf.random.uniform(shape=(bs,), maxval=N, dtype=tf.int32)  # bs
    neighbour_index_list = []  # bs, n_points, n_samples

    for i in range(n_points - 1):
        sample_mask = sample_mask + tf.one_hot(sample_id, depth=N)
        sample_id = tf.reshape(sample_id, shape=(-1, 1, 1))
        sampled_pt = tf.gather(xyz, sample_id, batch_dims=1)  # bs, 1 ,3
        relative_dist = tf.linalg.norm(tf.subtract(xyz, sampled_pt), axis=-1)  # bs, N
        _, neighbour_idx = tf.math.top_k(relative_dist, k=n_samples)
        # print(neighbour_idx.shape)
        # neighbour_idx = tf.expand_dims(neighbour_idx, 1)  # bs, 1, N
        neighbour_index_list.append(neighbour_idx)
        dist = tf.math.minimum(dist, relative_dist)
        sample_id = tf.reshape(tf.argmax(dist, axis=-1), shape=(-1,))  # bs,

    new_xyz = tf.boolean_mask(xyz, tf.cast(sample_mask, tf.bool))

    neighbour_index_list = tf.concat(neighbour_index_list, axis=1)
    return new_xyz, neighbour_index_list


@tf.function
def three_nn(xyz_source, xyz_target):
    """
    Up-interpolating features of xyz_source to xyz_target
    Building interpolation correspondences in xyz_source based on the distance, where n_pts_source < n_pts_target
    :param xyz_source: bs, n_pts_source, 3
    :param xyz_target: bs, n_pts_target, 3
    :return: dis, idx
    """

    k = 3
    bs, n_pts_target, c = xyz_target.shape
    xyz_source = tf.repeat(tf.expand_dims(xyz_source, axis=1), repeats=n_pts_target,axis=1)  # bs, n_pts_target, n_pts_source, 3
    xyz_target = tf.expand_dims(xyz_target, axis=2)  # bs, n_pts_target, 1, 3
    dis = tf.linalg.norm(tf.subtract(xyz_source, xyz_target), axis=-1)  # bs, n_pts_target, n_pts_source
    neighbour_dis, neighbour_idx = tf.math.top_k(-1 * dis, k=k)  # bs, n_pts_target, 3

    neighbour_dis = tf.maximum(-1 * neighbour_dis, 1e-10)
    norm = tf.reduce_sum((1.0 / neighbour_dis), axis=2, keepdims=True)
    norm = tf.tile(norm, [1, 1, 3])
    weight = (1.0 / neighbour_dis) / norm

    return weight, neighbour_idx


@tf.function
def three_interpolate(feats_source, idx, weights):
    """
    Interpolating features of xyz_source to xyz_target
    :param feats_source: the features from previous layer bs, n_pts_source, c
    :param idx: corresponding index in xyz_source
    :param weights: inverse distance weights (the farther the less important)
    :return: interpolated features bs, n_pts_target
    """
    weights_sum = tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-6  # bs, n_pts_target, 1
    weights_expand = tf.expand_dims(weights, axis=-1)  # bs, n_pts_target, 3, 1
    feats_selected = tf.gather(feats_source, indices=idx, batch_dims=1)  # bs, n_pts_target, 3, c
    inter_feats = tf.reduce_sum(weights_expand * feats_selected, axis=2) / weights_sum  # bs, n_pts_target, c
    return inter_feats


class PointNet2Params:
    def __init__(self):
        self.bn = False
        self.is_train = True
        self.keep_prob = 0.5
        self.return_features = True
        self.use_tf_interpolation = True
        self.n_sample_points = 12288


class PointNet2TfModel(Model):
    def __init__(self, params: PointNet2Params, num_classes):
        super(PointNet2TfModel, self).__init__()

        self.params = params
        self.activation = tf.nn.relu
        self.keep_prob = self.params.keep_prob
        self.num_classes = num_classes
        self.bn = self.params.bn

        self.kernel_initializer = 'glorot_normal'
        self.kernel_regularizer = None
        self.group_all = False

        self.init_network()

    def init_network(self):
        self.sa_1 = Pointnet_SA(
            npoint=512,  # todo we don't need that much
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

        return seg_features


class Pointnet_FP(Layer):

    def __init__(
            self, mlp, activation=tf.nn.relu, bn=False
    ):

        super(Pointnet_FP, self).__init__()

        self.mlp = mlp
        self.activation = activation
        self.bn = bn

        self.mlp_list = []

    def build(self, input_shape):

        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(SharedMlP(n_filters, activation=self.activation, bn=self.bn))
        super(Pointnet_FP, self).build(input_shape)

    def call(self, xyz_target, xyz_source, feats_target, feats_source, training=True):

        weight, neighbour_idx = three_nn(xyz_source, xyz_target)
        interpolated_feats = three_interpolate(feats_source, neighbour_idx, weight)

        if feats_target is not None:
            new_feats_target = tf.concat(axis=2,
                                         values=[interpolated_feats, feats_target])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_feats_target = interpolated_feats

        new_feats_target = tf.expand_dims(new_feats_target, 2)  # new feats bs, n_points_2, 1, feats

        for i, mlp_layer in enumerate(self.mlp_list):
            new_feats_target = mlp_layer(new_feats_target, training=training)

        new_feats_target = new_feats_target[:, :, 0, :]

        return new_feats_target


class SharedMlP(Layer):

    def __init__(self, filters, strides=[1, 1], activation=tf.nn.relu, padding='VALID', initializer='glorot_normal',
                 bn=False):

        super(SharedMlP, self).__init__()

        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.initializer = initializer
        self.bn = bn

    def build(self, input_shape):

        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name='pnet_conv'
        )

        if self.bn: self.bn_layer = BatchNormalization()

        super(SharedMlP, self).build(input_shape)

    def call(self, inputs, training=True):

        points = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)

        if self.bn: points = self.bn_layer(points, training=training)

        if self.activation: points = self.activation(points)

        return points
