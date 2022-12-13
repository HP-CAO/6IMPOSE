import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import write_config
from collections import OrderedDict
from lib.data.dataset_params import PvnDatasetParams
from lib.trainer.trainer_pvn import TrainerPvnParams, TrainerPvn3d
from lib.net.pvn3d import Pvn3dNet, Pvn3dNetParams
from lib.monitor.monitor import MonitorParams, Monitor
# from pvn3d.data.unity.unity_pvn import UnityPvn3d
from lib.data.dataset_params import pvn3d_tensor_spec
from lib.data.linemod.linemod_pvn import LineModPvn3d
# from pvn3d.data.unity.unity_settings import UnitySettings
from lib.data.linemod.linemod_settings import LineModSettings
import os
import argparse
from utils import read_config
from focal_loss import BinaryFocalLoss
from time import time


class MainPvn3dParams:
    def __init__(self):
        self.dataset_params = PvnDatasetParams()
        self.trainer_params = TrainerPvnParams()
        self.pvn3d_params = Pvn3dNetParams()
        self.monitor_params = MonitorParams()


class MainPvn3d:

    def __init__(self, params: MainPvn3dParams):

        self.params = params
        self.monitor = Monitor(self.params.monitor_params)
        write_config(params, f"{self.monitor.log_dir}/config.json")
        self.cls_type = self.params.dataset_params.cls_type
        self.crop_image = self.params.dataset_params.crop_image
        self.use_preprocessed = self.params.dataset_params.use_data_preprocessed

        if self.params.dataset_params.dataset == 'unity':
            # self.data_generator = UnityPvn3d
            # self.data_config = UnitySettings(
            #     self.params.dataset_params.data_name, self.cls_type, self.use_preprocessed, self.crop_image)
            pass

        elif self.params.dataset_params.dataset == 'linemod':
            self.data_generator = LineModPvn3d
            self.data_config = LineModSettings(
                self.params.dataset_params.data_name, self.cls_type, self.use_preprocessed, self.crop_image,
                self.params.dataset_params.size_all, self.params.dataset_params.train_size)

        self.is_binary = self.cls_type != 'all'

        self.train_dataset = self.data_from_generator(self.params.dataset_params.data_name, self.data_generator,
                                                      'train', self.params.dataset_params.train_batch_size,
                                                      self.params.dataset_params.size_all,
                                                      self.params.dataset_params.train_size)

    def train_loop(self, train_dataset, train_step):

        for epoch in range(self.params.monitor_params.train_epochs):
            tqdm.write(f"Training on epoch {epoch}")
            bar = tqdm(total=self.params.dataset_params.train_size, leave=False)
            for inputs in train_dataset:
                loss, loss_kp, loss_seg, loss_cp = train_step(inputs)
                # print("Loss: {} == Loss_kp: {} == Loss_seg: {} == Loss_cp: {}".format(loss, loss_kp, loss_seg, loss_cp))
                bar.update(self.params.dataset_params.train_batch_size)
                postfix = OrderedDict(loss=f"{loss}")
                bar.set_postfix(postfix)
            bar.close()

    def initial_trainer_and_model(self):

        pvn3d_model = Pvn3dNet(self.params.pvn3d_params,
                               rgb_input_shape=self.data_config.rgb_input_shape,
                               num_kpts=self.data_config.n_key_points,
                               num_cls=self.data_config.n_classes,
                               num_cpts=self.data_config.n_ctr_points,
                               dim_xyz=self.data_config.dim_pcld_xyz)
        return pvn3d_model

    def data_from_generator(self, data_name, data_generator, mode, batch_size, size_all, train_size):

        generator_args = \
            [mode, data_name, self.cls_type, self.data_config.use_preprocessed, size_all, train_size, self.crop_image]

        output_args = \
            pvn3d_tensor_spec(self.cls_type, self.data_config.rgb_input_shape, 12288, mode) # fixed -> sample online

        tf_ds = tf.data.Dataset.from_generator(data_generator, args=generator_args, output_signature=output_args).batch(
            batch_size).prefetch(tf.data.AUTOTUNE)

        return tf_ds


@tf.function
def distributed_train_step(dist_inputs):
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)


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


@tf.function
def loss_fn_pvn3d(kp_pre_ofst, kp_targ_ofst, seg_pre, cp_pre_ofst, ctr_targ_ofst,
                  label, mask_label, binary_loss=True):
    """
    joint loss function
    """
    bs, _, _, _ = kp_pre_ofst.shape

    loss_kp = l1_loss(pred_ofst=kp_pre_ofst,
                      targ_ofst=kp_targ_ofst,
                      mask_labels=mask_label)
    if binary_loss is True:
        seg_pre = tf.unstack(seg_pre, axis=2)[1]
        label = tf.argmax(label, axis=2)
        loss_seg = BinaryFocalLoss(label, seg_pre)  # retruen batch-wise value
    else:
        loss_seg = CategoricalCrossentropy(label, seg_pre)  # labels [bs, n_pts, n_cls] this is from logits

    loss_cp = l1_loss(pred_ofst=cp_pre_ofst,
                      targ_ofst=ctr_targ_ofst,
                      mask_labels=mask_label)

    loss_seg = tf.reduce_sum(loss_seg) * (1 / bs)

    loss = 1 * loss_kp + \
           1 * loss_seg + \
           1 * loss_cp

    return loss, loss_kp, loss_seg, loss_cp


@tf.function
def train_step(inputs):
    rgb, pcld_xyz, pcld_feats, sampled_index, label, kp_targ_ofst, ctr_targ_ofst, mask_label, crop_factor \
        = inputs

    with tf.GradientTape() as Tape:
        input_data = [rgb, pcld_xyz, pcld_feats, sampled_index]

        rgb, pcld_xyz, pcld_feats, sampled_index = input_data

        pcld = [pcld_xyz, pcld_feats]
        inputs = [pcld, sampled_index, rgb]

        kp_pre_ofst, seg_pre, cp_pre_ofst = pvn3d_model(inputs, training=True)

        loss, loss_kp, loss_seg, loss_cp = loss_fn_pvn3d(kp_pre_ofst=kp_pre_ofst,
                                                         kp_targ_ofst=kp_targ_ofst,
                                                         seg_pre=seg_pre,
                                                         cp_pre_ofst=cp_pre_ofst,
                                                         ctr_targ_ofst=ctr_targ_ofst,
                                                         label=label,
                                                         mask_label=mask_label,
                                                         binary_loss=True)
    grads = Tape.gradient(loss, pvn3d_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, pvn3d_model.trainable_variables))

    return loss, loss_kp, loss_seg, loss_cp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/pvn3d_linemod_gpu_test.json', help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--force', action='store_true', help='Override log file without asking')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    # './model/model_name/model_name'
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--mode', default='train', help='Choose the mode, train or test')
    parser.add_argument('--net', default="pvn", help="Running different network: pvn, pvn_kp, pvn_adp")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    if args.config is None:
        exit("config file needed")

    params = read_config(args.config)

    if args.id is not None:
        params.monitor_params.model_name = args.id
        params.monitor_params.log_file_name = args.id

    params.monitor_params.force_override = True

    params.monitor_params.weights_path = args.weights

    params.monitor_params.mode = args.mode

    network = MainPvn3d(params)

    strategy = \
        tf.distribute.MirroredStrategy(devices=params.trainer_params.distribute_train_device,
                                       cross_device_ops=tf.distribute.ReductionToOneDevice())

    train_dataset = network.train_dataset
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = strategy.experimental_distribute_dataset(train_dataset.with_options(options))
    reduction = tf.keras.losses.Reduction.NONE
    BinaryFocalLoss = BinaryFocalLoss(gamma=2, from_logits=False, reduction=reduction)
    CategoricalCrossentropy = \
        tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=reduction)

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        pvn3d_model = network.initial_trainer_and_model()

    warm_up = True

    for epoch in range(10):
        tqdm.write(f"Training on epoch {epoch}")
        bar = tqdm(total=params.dataset_params.train_size, leave=False)
        time_list = []
        if warm_up:
            counter = 0
            print(
                "===== current using {} Gpus =====".format(len(params.trainer_params.distribute_train_device)))
            for inputs in train_dataset:
                t_start_warm = time()
                distributed_train_step(inputs)
                print("warming up, time: {}".format(time() - t_start_warm))
                warm_up = False
                counter += 1
                if counter == 20:
                    break

        for inputs in train_dataset:
            t0 = time()
            dis_loss, _, _, _ = distributed_train_step(inputs)
            bar.update(params.dataset_params.train_batch_size)
            postfix = OrderedDict(loss=f"{dis_loss}")
            bar.set_postfix(postfix)
            delta_t = time() - t0
            print("time for one batch {} is {} ".format(params.dataset_params.train_batch_size, delta_t))
            time_list.append(delta_t)

        mean_t_per_batch = np.array(time_list).mean()
        print("mean_t_per_batch is: {} speed is: "
              "{} images/s".format(mean_t_per_batch, params.dataset_params.train_batch_size / mean_t_per_batch))

        bar.close()

