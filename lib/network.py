import os
import tensorflow as tf
from abc import ABC, abstractmethod
from lib.data.dataset_settings import DatasetSettings
from typing import Callable, List, Tuple, Dict
from tqdm import tqdm
from collections import OrderedDict
from lib.monitor.monitor import Monitor
from lib.trainer.trainer import Trainer
from utils import write_config
from tensorflow.keras import Model
from lib.params import NetworkParams


class Network(ABC):
    data_config: DatasetSettings
    data_generator: Callable  # returns Dataset
    params: NetworkParams

    train_generator_args: List
    val_generator_args: List
    train_tensor_spec: Tuple
    val_tensor_spec: Tuple

    model: Model
    trainer: Trainer

    @abstractmethod
    def initial_trainer_and_model(self):
        """ instanciate trainer and model"""
        pass

    @abstractmethod
    def train_step(self, inputs)->Dict[str, float]:
        """ train and optimize, return dict with losses
            containing at least 'loss' == overall loss """
        pass

    @abstractmethod
    def val_step(self, inputs):
        pass

    @abstractmethod
    def performance_evaluation(self, epoch):
        pass

    def pre_training(self):
        pass

    def run_demo(self):
        pass

    # -------------------------------------------------------------------------------------------------------
    strategy = None

    def __init__(self, params: NetworkParams):
        self.params = params
        self.monitor = Monitor(self.params.monitor_params)
        if params.monitor_params.write_log:
            write_config(params, f"{self.monitor.log_dir}/config.json")

        self.global_train_batch_size = self.params.dataset_params.train_batch_size
        self.global_val_batch_size = self.params.dataset_params.val_batch_size

        from lib.factory import DatasetFactory
        self.factory = DatasetFactory(self.params)
        self.data_config = self.factory.get_data_config()

    def train(self):
        if self.params.trainer_params.distribute_training:
            num_gpus = len(self.params.trainer_params.distribute_train_device)
            self.global_train_batch_size = int(self.params.dataset_params.train_batch_size * num_gpus)
            self.global_val_batch_size = int(self.params.dataset_params.val_batch_size * num_gpus)

        # data_from_tfrecord, data_from_generator
        if self.data_config.use_preprocessed:
            train_dataset = self.factory.data_from_tfrecord('train').batch(self.global_train_batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            train_dataset = self.factory.data_from_generator('train').batch(self.global_train_batch_size).prefetch(tf.data.AUTOTUNE)

        if self.params.monitor_params.if_validation:
            val_dataset = self.factory.data_from_generator('val').batch(self.global_val_batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            val_dataset = None

        if self.params.trainer_params.distribute_training:
            self.strategy = \
                tf.distribute.MirroredStrategy(devices=self.params.trainer_params.distribute_train_device,
                                               cross_device_ops=tf.distribute.ReductionToOneDevice())

            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset.with_options(options))
            val_dataset = self.strategy.experimental_distribute_dataset(val_dataset.with_options(options))

            with self.strategy.scope():
                self.initial_trainer_and_model()

        else:
            self.initial_trainer_and_model()

        self.train_loop(train_dataset, val_dataset)

    @tf.function
    def distributed_train_step(self, dist_inputs):
        per_replica_losses = self.strategy.run(self.train_step, args=(dist_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    @tf.function
    def distributed_val_step(self, dist_inputs):
        per_replica_losses = self.strategy.run(self.val_step, args=(dist_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    def train_loop(self, train_dataset, val_dataset):
        self.pre_training()

        train_len = self.factory.get_len('train')
        val_len = self.factory.get_len('val')

        for epoch in range(self.params.monitor_params.train_epochs):

            self.trainer.reset()

            tqdm.write(f"Training on epoch {epoch}")
            bar = tqdm(total=train_len)

            for inputs in train_dataset:

                if self.strategy is None:
                    losses = self.train_step(inputs)
                else:
                    losses = self.distributed_train_step(inputs)
                self.trainer.log(losses)

                bar.update(self.global_train_batch_size)
                postfix = OrderedDict(loss=f"{losses['loss'].numpy():.3f}")
                bar.set_postfix(postfix)
            bar.close()

            self.monitor.train_summary(epoch, self.trainer.get())

            if self.monitor.is_val(epoch, is_validation=self.params.monitor_params.if_validation):

                self.trainer.reset()

                tqdm.write(f"Val on epoch {epoch}")
                bar = tqdm(total=int(val_len))

                for val_inputs in val_dataset:
                    if self.strategy is None:
                        losses = self.val_step(val_inputs)
                    else:
                        losses = self.distributed_val_step(val_inputs)

                    self.trainer.log(losses)

                    bar.update(self.global_val_batch_size)
                    postfix = OrderedDict(loss=f"{losses['loss'].numpy():.3f}")
                    bar.set_postfix(postfix)

                bar.close()
                self.monitor.val_summary(epoch, self.trainer.get())

                val_loss = self.trainer.get_overall_loss()

                if self.monitor.is_performance_eval(epoch):
                    self.performance_evaluation(epoch)

                if self.monitor.is_save_checkpoints(val_loss):
                    model_save_name = '/best_model/model'
                    self.model.save_weights(self.monitor.save_model_dir + model_save_name)
            
            if epoch % self.params.monitor_params.model_save_period == 0:
                model_save_name = f'/{epoch}/model'
                self.model.save_weights(self.monitor.save_model_dir + model_save_name)
    
    def export_model(self):
        self.model.save(os.path.join(self.monitor.save_model_dir, 'exported'))
