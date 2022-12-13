import os
import shutil
import distutils.util
from typing import List
import numpy as np
import tensorflow as tf
from lib.monitor.evaluator import EvalResult, EvalResultType

from lib.params import MonitorParams


class Monitor:
    def __init__(self, params: MonitorParams):
        self.params = params

        self.log_dir = self.params.log_root + self.params.model_name
        self.resnet_pre_path = os.path.join(self.params.model_dir, self.params.resnet_weights_name)
        self.log_train_dir = self.log_dir + "/train"
        self.log_val_dir = self.log_dir + "/val"
        self.log_test_dir = self.log_dir + "/test"
        self.save_model_dir = os.path.join(self.params.model_dir, self.params.model_name)
        self.best_loss = self.params.best_loss
        if params.write_log:
            self.clear_cache()
            self.create_tensorboard_writer()
        self.add_list = []
        self.adds_list = []

    def create_tensorboard_writer(self):
        if self.params.mode == "train":
            self.train_log_writer = tf.summary.create_file_writer(self.log_train_dir)
            self.val_log_writer = tf.summary.create_file_writer(self.log_val_dir)
        else:
            self.test_log_writer = tf.summary.create_file_writer(self.log_test_dir)

    def model_graph_summary(self, model):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=100)
        tensorboard_callback.set_model(model)

    def write_loss_scalar(self, epoch, loss_values, mode):
        for name, val in loss_values.items():
            tf.summary.scalar(f"{name}", val, epoch)
        #print("{}>> Epoch-{}: loss: {}".format(mode, epoch, loss_values['loss']))

    def train_summary(self, epoch, loss_values):
        with self.train_log_writer.as_default():
            self.write_loss_scalar(epoch, loss_values, 'Train')

    def val_summary(self, epoch, loss_values):
        with self.val_log_writer.as_default():
            self.write_loss_scalar(epoch, loss_values, 'Val')

    def log_eval_results(self, epoch, eval_results: List[EvalResult], name):
        with self.val_log_writer.as_default():
            for eval_result in eval_results:
                if eval_result.type == EvalResultType.images:
                    tf.summary.image(f"{name}_{eval_result.name}/", np.array(eval_result.data), max_outputs=len(eval_result.data), step=epoch)
                elif eval_result.type == EvalResultType.scalar:
                    tf.summary.scalar(f"{name}_{eval_result.name}", eval_result.data, step=epoch)
                else:
                    print(f"Unknown performance result: {eval_result}")

    def __performance_eval_pt2(self, epoch, eval_result, name):
        iou_mean, gt_pre_seg_list = eval_result
        num_samples = len(gt_pre_seg_list)

        with self.val_log_writer.as_default():
            tf.summary.scalar('{}_performance/iou_mean'.format(name), iou_mean, epoch)

            for i in range(num_samples):
                tf.summary.image("{}_{}_gt_pre/gt_pre".format(name, i), np.array(gt_pre_seg_list[i]),
                                 max_outputs=len(gt_pre_seg_list[i]), step=epoch)

    def is_val(self, epoch, is_validation):
        if not is_validation:
            return False

        return True if epoch % self.params.val_frequency == 0 else False

    def is_performance_eval(self, epoch):

        return True if epoch % (
                self.params.val_frequency * self.params.performance_eval_frequency_factor) == 0 else False

    def is_save_checkpoints(self, val_loss):

        if val_loss <= self.best_loss:
            self.best_loss = val_loss
            return True
        else:
            return False

    def clear_cache(self):
        if os.path.isdir(self.log_dir):
            if self.params.force_override:
                shutil.rmtree(self.log_dir)
            else:
                print(self.log_dir, 'already exists.')
                resp = input('Override log file? [Y/n]\n')
                if resp == '' or distutils.util.strtobool(resp):
                    print('Deleting old log dir')
                    shutil.rmtree(self.log_dir)
                else:
                    print('Okay bye')
                    exit(1)
