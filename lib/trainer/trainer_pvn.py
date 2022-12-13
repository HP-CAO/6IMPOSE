import numpy as np
import tensorflow as tf
from focal_loss import BinaryFocalLoss
from lib.trainer.trainer import Trainer, TrainerParams
from typing import Dict


class TrainerPvnParams(TrainerParams):
    def __init__(self):
        self.distribute_training = False
        self.distribute_train_device = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"]
        self.focal_loss_gamma = 2
        self.focal_loss_alpha = None
        self.reduce_loss_mean = True
        self.learning_rate = 1e-3
        self.kp_loss_discount = 1.0
        self.sm_loss_discount = 0.01
        self.cp_loss_discount = 1.0
        self.kp_cp_loss_discount = 1
        self.kp_cp_ofst_loss_discount = 1
        self.seg_from_logits = True


class TrainerPvn3d(Trainer):
    def __init__(self, params: TrainerPvnParams):
        super().__init__()
        self.params = params
        self.reduction = tf.keras.losses.Reduction.NONE if self.params.distribute_training else tf.keras.losses.Reduction.AUTO
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        self.BinaryFocalLoss = BinaryFocalLoss(gamma=2, from_logits=self.params.seg_from_logits, reduction=self.reduction)
        self.CategoricalCrossentropy = \
            tf.keras.losses.CategoricalCrossentropy(from_logits=self.params.seg_from_logits, reduction=self.reduction)

        self.loss_list = []
        self.loss_kp_list = []
        self.loss_cp_list = []
        self.loss_seg_list = []

    def reset(self):
        self.loss_list = []
        self.loss_kp_list = []
        self.loss_cp_list = []
        self.loss_seg_list = []

    def log(self, losses:Dict[str, float]):
        self.loss_list.append(losses['loss'])
        self.loss_kp_list.append(losses['loss_kp'])
        self.loss_cp_list.append(losses['loss_cp'])
        self.loss_seg_list.append(losses['loss_seg'])

    def get_overall_loss(self) -> float:
        return np.array(self.loss_list).mean()

    def get(self)->Dict[str, float]:
        mean_loss = np.array(self.loss_list).mean()
        mean_loss_kp = np.array(self.loss_kp_list).mean()
        mean_loss_cp = np.array(self.loss_cp_list).mean()
        mean_loss_seg = np.array(self.loss_seg_list).mean()
        return {'pvn_loss/loss':mean_loss, 'pvn_loss/loss_kp':mean_loss_kp, 'pvn_loss/loss_seg':mean_loss_seg, 'pvn_loss/loss_cp':mean_loss_cp}

    @tf.function
    def loss_fn_pvn3d(self, kp_pre_ofst, kp_targ_ofst, seg_pre, cp_pre_ofst, ctr_targ_ofst,
                      label, mask_label, binary_loss=True):
        """
        joint loss function
        """
        bs, _, _, _ = kp_pre_ofst.shape

        loss_kp = self.l1_loss(pred_ofst=kp_pre_ofst,
                               targ_ofst=kp_targ_ofst,
                               mask_labels=mask_label)
        if binary_loss is True:

            if not self.params.seg_from_logits:
                seg_pre = tf.nn.softmax(seg_pre)

            seg_pre = tf.unstack(seg_pre, axis=2)[1]
            label = tf.argmax(label, axis=2)
            loss_seg = self.BinaryFocalLoss(label, seg_pre)  # return batch-wise value
        else:
            loss_seg = self.CategoricalCrossentropy(label, seg_pre)  # labels [bs, n_pts, n_cls] this is from logits

        loss_cp = self.l1_loss(pred_ofst=cp_pre_ofst,
                               targ_ofst=ctr_targ_ofst,
                               mask_labels=mask_label)

        if self.params.distribute_training:
            loss_seg = tf.reduce_sum(loss_seg) * (1 / bs)

        loss_cp = self.params.cp_loss_discount * loss_cp
        loss_kp = self.params.kp_loss_discount * loss_kp
        loss_seg = self.params.sm_loss_discount * loss_seg

        loss = loss_cp + loss_kp + loss_seg

        return loss, loss_kp, loss_seg, loss_cp

    def loss_pvn_ktps(self, kp_cp_ofst_pre, kp_cp_ofst_target, kp_cp_pre, kp_cp_target, seg_pre, label, mask_label):
        """
        :param kp_cp_ofst_pre:
        :param kp_cp_ofst_target:
        :param kp_cp_pre:
        :param kp_cp_target:
        :param seg_pre: One-shot encoding for pixel-wise semantics
        :param label:
        :param mask_label:
        :param target_cls:
        :return:
        """
        loss_kp_cp_ofst = self.l1_loss(pred_ofst=kp_cp_ofst_pre, targ_ofst=kp_cp_ofst_target, mask_labels=mask_label)
        seg_pre = tf.unstack(seg_pre, axis=2)[1]
        label = tf.argmax(label, axis=2)
        loss_seg = self.BinaryFocalLoss(label, seg_pre)  # return batch-wise loss
        loss_kp_cp = self.l1_loss_kp_cp(kp_cp_pre, kp_cp_target)

        loss = self.params.kp_cp_ofst_loss_discount * loss_kp_cp_ofst + \
               self.params.sm_loss_discount * loss_seg + \
               self.params.kp_cp_loss_discount * loss_kp_cp

        return loss, loss_kp_cp_ofst, loss_seg, loss_kp_cp

    @staticmethod
    def l1_loss_kp_cp(kp_cp_pre, kp_cp_targ):
        diffs = tf.subtract(kp_cp_pre, kp_cp_targ)
        abs_diff = tf.math.abs(diffs)
        l1_loss_kp_cp = tf.reduce_mean(abs_diff)
        return l1_loss_kp_cp

    @staticmethod
    @tf.function
    def l1_loss(pred_ofst, targ_ofst, mask_labels):
        """
        :param: pred_ofsts: [bs, n_pts, n_kpts, c] or [bs, n_pts, n_cpts, c]
                targ_ofst: [bs, n_pts, n_kpts, c] for kp,  [bs, n_pts, n_cpts, c] for cp
                mask_labels: [bs, n_pts]
        """
        bs, n_pts, n_kpts, c = pred_ofst.shape
        num_nonzero = tf.cast(tf.math.count_nonzero(mask_labels), tf.float32)

        w = tf.cast(mask_labels, dtype=tf.float32)
        w = tf.reshape(w, shape=[bs, n_pts, 1, 1])
        w = tf.repeat(w, repeats=n_kpts, axis=2)

        diff = tf.subtract(pred_ofst, targ_ofst)
        abs_diff = tf.multiply(tf.math.abs(diff), w)
        in_loss = abs_diff
        l1_loss = tf.reduce_sum(in_loss) / (num_nonzero + 1e-3)

        return l1_loss
