from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from ladder_network import tensor_utils
from ladder_network.wrappers import Registry


initializers = Registry('Initialization')


@initializers.register('glorot_uniform')
def glorot_uniform(weight_shape, dtype=tf.float32, partition_info=None, seed=42):
    """Glorot uniform weight initialization"""
    n_inputs, n_outputs = tensor_utils.get_fans(weight_shape)
    init_range = np.sqrt(6. / (n_inputs + n_outputs))
    return tf.random_uniform(
        weight_shape, -init_range, init_range, seed=seed, dtype=dtype)


@initializers.register('glorot_normal')
def glorot_normal(weight_shape, dtype=tf.float32, partition_info=None, seed=42):
    """Glorot normal weight initialization"""
    n_inputs, n_outputs = tensor_utils.get_fans(weight_shape)
    stddev = np.sqrt(2. / (n_inputs + n_outputs))
    return tf.truncated_normal(
        weight_shape, 0.0, stddev=stddev, seed=seed, dtype=dtype)


@initializers.register('he_uniform')
def he_uniform(weight_shape, dtype=tf.float32, partition_info=None, seed=42):
    """He uniform weight initialization."""
    n_inputs, n_outputs = tensor_utils.get_fans(weight_shape)
    init_range = np.sqrt(6. / n_inputs)
    return tf.random_uniform(
        weight_shape, -init_range, init_range, seed=seed, dtype=dtype)


@initializers.register('he_normal')
def he_normal(weight_shape, dtype=tf.float32, partition_info=None, seed=42):
    """He normal weight initialization."""
    n_inputs, n_outputs = tensor_utils.get_fans(weight_shape)
    stddev = np.sqrt(2. / n_inputs)
    return tf.truncated_normal(
        weight_shape, 0.0, stddev=stddev, seed=seed, dtype=dtype)
