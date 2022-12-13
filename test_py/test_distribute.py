import tensorflow as tf
from tensorflow.keras import Model
from focal_loss import BinaryFocalLoss

import numpy as np
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Input, \
    SpatialDropout2D, PReLU, Concatenate, ReLU, Flatten


def mnist_tensor_spec_test():

    tensor_spec = (
        tf.TensorSpec(shape=(28, 28), dtype=tf.float32, name='input'),
        tf.TensorSpec(shape=1, dtype=tf.float32, name='label'),
    )
    return tensor_spec


class MnistGenerator:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.len_dataset = 2000
        self.load_dataset()
        self.counter = 0

    def load_dataset(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train / 255.0
        self.x_train = x_train
        self.y_train = y_train

    def get_item(self):
        return self.x_train[self.counter], [self.y_train[self.counter]]

    def has_next(self):
        return self.counter < self.len_dataset

    def next(self):
        item = self.get_item()
        self.counter += 1
        return item

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            raise StopIteration()


@tf.function  # this has to be decorated
def train_step(inputs):
    x_train, y_train = inputs
    with tf.GradientTape() as Tape:
        predictions = model(x_train, training=True)
        pre = tf.nn.softmax(predictions)
        loss = loss_fn(y_train, pre)
        loss = tf.math.divide(loss, 12)
        # print("loss: ", loss)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=50)
        # print("loss_compute_mean:", loss)
    grads = Tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# @tf.function
def distribute_train_step(inputs):
    per_replica_losses = strategy.run(train_step, args=(inputs,))
    # print("per_replica_losses:", per_replica_losses)
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def loss_fn(y_train, pre, cross_loss=False):
    bs_shape = y_train.shape[0]
    if not cross_loss:
        loss = loss_function_bi(y_train, pre)
    else:
        loss = loss_function_cross(y_train, pre)
    loss = tf.math.divide(loss, bs_shape)
    loss = tf.math.abs(loss)
    loss = tf.math.log(loss)
    return loss


class my_model(Model):

    def __init__(self):
        super(my_model, self).__init__()
        self.layer1 = tf.keras.layers.Flatten()
        self.layer2 = tf.keras.layers.Dense(128, activation="relu")
        self.layer3 = tf.keras.layers.Dropout(0.2)
        self.layer4 = tf.keras.layers.Dense(1)
        print("===Model created via subclass===")

    def call(self, inputs, training=None, mask=None):
        x = inputs
        # x = self.input_layer(inputs)
        # x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def build_model():
    input_layer = tf.keras.layers.Input(shape=(28, 28))
    layer1 = tf.keras.layers.Flatten(input_shape=(28, 28))(input_layer)
    layer2 = tf.keras.layers.Dense(128, activation="relu")(layer1)
    layer3 = tf.keras.layers.Dropout(0.2)(layer2)
    output = tf.keras.layers.Dense(1)(layer3)
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    print("===Model created via functional API===")
    return model


strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],
                                          cross_device_ops=tf.distribute.ReductionToOneDevice())
with strategy.scope():
    model = build_model()
    # model = my_model()
    optimizer = tf.keras.optimizers.Adam()

train_dataset = tf.data.Dataset.from_generator(MnistGenerator, output_signature=mnist_tensor_spec_test(
)).batch(batch_size=50).prefetch(tf.data.AUTOTUNE)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dataset = strategy.experimental_distribute_dataset(train_dataset.with_options(options))
loss_function_cross = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                              reduction=tf.keras.losses.Reduction.NONE)

loss_function_bi = BinaryFocalLoss(gamma=2, from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

for i in range(5):
    for dist_inputs in train_dataset:
        print("Distribute reduce mean: ", distribute_train_step(dist_inputs))
