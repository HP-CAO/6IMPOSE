import os
from charset_normalizer import detect

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/local/cuda/"

# import silence_tensorflow.auto

# include submodule into module so pvn3d can be found
import sys

sys.path.append(os.path.join("networks", "pvn"))

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
except Exception as e:
    print("Error setting gpu memory growth: ", e)
    # Invalid device or cannot modify virtual devices once initialized.
    pass

tf.config.optimizer.set_experimental_options({"debug_stripper": True})

tf.config.optimizer.set_jit('autocluster')

import cv2
import shelve
import time
import os
import begin
import numpy as np