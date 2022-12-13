import tensorflow as tf
from tensorflow.keras import Model
from lib.geometry.geometry import batch_rt_svd_transform, batch_pts_clustering_with_std, batch_get_pt_candidates_tf


class PProcessParams:
    def __init__(self):
        self.use_stochastic_icp = True
        self.n_point_candidate = 10
        self.distribution = "normal"
        self.angle_bound = 0.15


class InitialPoseModel(Model):
    def __init__(self, n_point_candidate=10):
        super(InitialPoseModel, self).__init__()
        self.n_point_candidates = n_point_candidate

    def call(self, inputs, training=False, mask=None):
        pcld_input, kpts_pre_input, cpt_pre_input, seg_pre_input, mesh_kpts_input = inputs

        obj_kpts = batch_get_pt_candidates_tf(pcld_input, kpts_pre_input, seg_pre_input,
                                              cpt_pre_input, self.n_point_candidates)

        kpts_voted = batch_pts_clustering_with_std(obj_kpts)
        _, n_pts, _ = kpts_voted.shape
        weights_vector = tf.ones(shape=(n_pts,))
        batch_R, batch_t = batch_rt_svd_transform(mesh_kpts_input, kpts_voted, weights_vector)
        batch_t = tf.reshape(batch_t, shape=(-1, 3))  # reshape from [bs, 3, 1] to [bs, 3]
        return batch_R, batch_t, kpts_voted


class TemporalInitialPoseModel(Model):
    def __init__(self, n_point_candidate=10):
        super().__init__()
        self.n_point_candidates = n_point_candidate
        self.last_kpts = None
        self.compl_factor = 0.4

    def call(self, inputs, training=False, mask=None):
        pcld_input, kpts_pre_input, cpt_pre_input, seg_pre_input, mesh_kpts_input = inputs

        obj_kpts = batch_get_pt_candidates_tf(pcld_input, kpts_pre_input, seg_pre_input,
                                              cpt_pre_input, self.n_point_candidates)

        kpts_voted = batch_pts_clustering_with_std(obj_kpts)
        _, n_pts, _ = kpts_voted.shape # [1,9,3]

        if self.last_kpts is not None:
            filtered_kpts_voted = self.compl_factor * self.last_kpts + (1-self.compl_factor) * kpts_voted
        else:
            filtered_kpts_voted = kpts_voted

        self.last_kpts = filtered_kpts_voted

        weights_vector = tf.ones(shape=(n_pts,))
        batch_R, batch_t = batch_rt_svd_transform(mesh_kpts_input, filtered_kpts_voted, weights_vector)
        batch_t = tf.reshape(batch_t, shape=(-1, 3))  # reshape from [bs, 3, 1] to [bs, 3]
        return batch_R, batch_t, filtered_kpts_voted
