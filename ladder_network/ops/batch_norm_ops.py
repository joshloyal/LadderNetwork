import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

import ladder_network.ops.training as training_ops
from ladder_network import tensor_utils


@slim.add_arg_scope
def scale_and_center(tensor_in,
                     scale=True,
                     center=True,
                     reuse=None,
                     variables_collections=None,
                     scope=None):
    """Applies the trainable batch-norm correction to a normalized tensor:
        u = gamma * (u_pre + beta)
    """
    with tf.variable_scope(scope, 'scale_and_offset', [tensor_in],
                           reuse=reuse) as sc:
        tensor_in = tf.convert_to_tensor(tensor_in)
        input_shape = tensor_utils.get_shape(tensor_in)

        outputs = tensor_in
        if center:
            beta_collections = layers.utils.get_variable_collections(
                variables_collections, "beta")
            beta = slim.variable("beta",
                                 shape=[input_shape[-1]],
                                 initializer=tf.zeros_initializer,
                                 collections=beta_collections,
                                 trainable=True)
            outputs = outputs + beta

        if scale:
            gamma_collections = layers.utils.get_variable_collections(
                variables_collections, "gamma")
            gamma = slim.variable("gamma",
                                  shape=[input_shape[-1]],
                                  initializer=tf.constant_initializer(1.),
                                  collections=gamma_collections,
                                  trainable=True)
            outputs = gamma * outputs

        return outputs


@slim.add_arg_scope
def batch_normalization(tensor_in,
                        epsilon=1e-10,
                        decay=0.9,
                        variables_collections=None,
                        outputs_collections=None,
                        reuse=None,
                        scope=None):
    """Element-wise batch normalization. This is only the first
    half of the typical batch normalization calculation
    (standardization by the batch mean and variance).
        u = (u_pre - mean) / variance
    """
    with tf.variable_scope(scope, 'batch_normalization', [tensor_in],
                           reuse=reuse) as sc:
        tensor_in = tf.convert_to_tensor(tensor_in)
        input_shape = tensor_in.get_shape().as_list()
        input_ndim = len(input_shape)
        axis = list(range(input_ndim - 1))

        moving_mean_collections = layers.utils.get_variable_collections(
            variables_collections, 'moving_mean')
        moving_mean = slim.variable('moving_mean',
                                    shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer,
                                    collections=moving_mean_collections,
                                    trainable=False)
        moving_variance_collections = layers.utils.get_variable_collections(
            variables_collections, "moving_variance")
        moving_variance = slim.variable('moving_variance',
                                        shape=input_shape[-1:],
                                        initializer=tf.constant_initializer(1.),
                                        collections=moving_variance_collections,
                                        trainable=False)

        def update_mean_var():
            mean, variance = tf.nn.moments(tensor_in, axis, name='moments')
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            with tf.control_dependencies([update_moving_mean,
                                          update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)

        is_training = training_ops.get_training_mode()
        mean, variance = control_flow_ops.cond(
                is_training,
                update_mean_var,
                lambda: (moving_mean, moving_variance))

        layers.utils.collect_named_outputs(
            outputs_collections, scope + '/mean', mean)

        layers.utils.collect_named_outputs(
            outputs_collections, scope + '/variance', variance)

        # actually apply the normalization
        variance_epsilon = tensor_utils.to_tensor(epsilon, tensor_in.dtype.base_dtype)
        return (tensor_in - mean) * tf.rsqrt(variance + variance_epsilon)
