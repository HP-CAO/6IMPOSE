import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D


class MlpClusterNetParams:
    def __init__(self):
        self.target_cls_shape = 1
        self.num_rgbd_feats = 1024
        self.kp_conv1d_1_dim = 1024
        self.kp_conv1d_2_dim = 512
        self.kp_conv1d_3_dim = 256
        self.seg_conv1d_1_dim = 1024
        self.seg_conv1d_2_dim = 512
        self.seg_conv1d_3_dim = 128
        self.pts_vote_dense_1_dim = 1024
        self.pts_vote_dense_2_dim = 512


class MlpClusterNet:
    def __init__(self, params: MlpClusterNetParams, num_pts=12228, num_kpts=9, num_cls=7, num_cpts=1, channel_xyz=3):
        """
        Build semantics and key-points prediction layer using shared MLPs
        :param params: The neural network parameters
        :param num_pts: the number of points candidates
        :param num_kpts: the number of key points (not including center points)
        :param num_cls: the number of classes in the dataset
        :param num_cpts:  = 1, the number of the center points of the object
        :param channel_xyz: = 3 the dim of the point (x,y,z)
        :param target_cls: the cls of the object detected by the yolo tiny
        """
        self.params = params
        self.num_pts = num_pts
        self.num_kp_cp = num_cpts + num_kpts
        self.num_cls = num_cls
        self.channel_xyz = channel_xyz

    def kp_cp_layers(self, rgbd_features):
        conv1d_1 = Conv1D(
            filters=self.params.kp_conv1d_1_dim, kernel_size=1, activation='relu', name="kp_conv1d_1")(rgbd_features)
        conv1d_2 = Conv1D(
            filters=self.params.kp_conv1d_2_dim, kernel_size=1, activation='relu', name="kp_conv1d_2")(conv1d_1)
        conv1d_3 = Conv1D(
            filters=self.params.kp_conv1d_3_dim, kernel_size=1, activation='relu', name="kp_conv1d_3")(conv1d_2)
        kp_cp_pre_feats = Conv1D(
            filters=self.num_kp_cp * self.channel_xyz, kernel_size=1, activation='relu', name="kp_cp_pre_feats")(conv1d_3) # [Bs, n_pts, num_kp_cp * channel_xyz]

        kp_cp_ofst = tf.reshape(kp_cp_pre_feats, shape=(-1, self.num_pts, self.num_kp_cp, self.channel_xyz), name='features reshape', )

        return kp_cp_ofst

    def seg_layers(self, rgbd_features):
        conv1d_1 = Conv1D(
            filters=self.params.seg_conv1d_1_dim, kernel_size=1, activation='relu', name="seg_conv1d_1")(rgbd_features)
        conv1d_2 = Conv1D(
            filters=self.params.seg_conv1d_2_dim, kernel_size=1, activation='relu', name="seg_conv1d_2")(conv1d_1)
        conv1d_3 = Conv1D(
            filters=self.params.seg_conv1d_3_dim, kernel_size=1, activation='relu', name="seg_conv1d_3")(conv1d_2)
        seg_pre = Conv1D(
            filters=self.num_cls, kernel_size=1, activation=None, name="seg_conv1d_4")(conv1d_3) # [Bs, n_pts, num_cls]

        return seg_pre

    def points_candidates_filter(self, kp_cp_ofst, pcld_xyz, seg_pre, target_cls):
        bs, n_pts, n_kp_cp, c = kp_cp_ofst.shape
        pcld_xyz = tf.expand_dims(pcld_xyz, axis=2)
        pcld_xyz = tf.repeat(pcld_xyz, repeats=self.num_kp_cp, axis=2)
        kp_cp_candidates = tf.add(kp_cp_ofst, pcld_xyz)
        segs_pb = tf.nn.softmax(seg_pre)
        segs = tf.argmax(segs_pb, axis=-1)
        segs_mask = tf.cast(segs == target_cls, dtype=tf.float32)
        segs_mask = tf.expand_dims(segs_mask, -1)
        segs_mask = tf.expand_dims(segs_mask, -1)

        segs_mask = tf.repeat(segs_mask, repeats=n_kp_cp, axis=2)
        segs_mask = tf.repeat(segs_mask, repeats=c, axis=3)

        kp_cp_cls_filtered = tf.multiply(kp_cp_candidates, segs_mask) # [Bs, n_pts, n_kp_cp, 3]
        kp_cp_cls_trans = tf.transpose(kp_cp_cls_filtered, perm=(0, 2, 1, 3)) # [Bs, num_kp_cp, n_pts, 3]

        kp_cp_cls_trans_reshape = tf.reshape(kp_cp_cls_trans, shape=(-1, self.num_kp_cp, self.num_pts * self.channel_xyz)) # [Bs, num_kp_cp, n_pts * channel_xyz]
        return kp_cp_cls_trans_reshape

    def pts_voting_layer(self, kp_cp_cls_trans_reshape):
        """
        Regress the all points to the single point
        :param kp_cp_cls_trans_reshape: [Bs, num_kp_cp, n_pts_filtered * 3]
        :return: pts_voted [bs, num_kp_cp, 3]
        """
        pts_vote_1 = Conv1D(
            self.params.pts_vote_dense_1_dim, kernel_size=1, activation="relu", name="points_voted_1")(kp_cp_cls_trans_reshape)
        pts_vote_2 = Conv1D(
            self.params.pts_vote_dense_2_dim, kernel_size=1, activation="relu", name="points_voted_2")(pts_vote_1)
        pts_voted = Conv1D(
            self.channel_xyz, kernel_size=1, activation=None, name="points_voted")(pts_vote_2)  # [Bs, num_kp_cp, 3]
        return pts_voted

    def build_mlp_cluster_model(self, rgbd_features_shape):
        input_rgbd_features = Input(shape=rgbd_features_shape, name='rgbd_features_input')
        input_target_cls = Input(shape=self.params.target_cls_shape, name='target_cls_input', dtype=tf.int64)

        input_points_xyz = Input(shape=(self.num_pts, self.channel_xyz), name='pcld_xyz_input')

        kp_cp_ofst = self.kp_cp_layers(input_rgbd_features)

        sm_pre_feats = self.seg_layers(input_rgbd_features)

        kp_cp_cls_trans_reshape = self.points_candidates_filter(kp_cp_ofst=kp_cp_ofst, pcld_xyz=input_points_xyz,
                                                                seg_pre=sm_pre_feats, target_cls=input_target_cls)

        pts_voted = self.pts_voting_layer(kp_cp_cls_trans_reshape)
        model = Model(inputs=[input_rgbd_features, input_target_cls, input_points_xyz],
                      outputs=[kp_cp_ofst, sm_pre_feats, pts_voted], name="kpts_cluster_model")

        return model

