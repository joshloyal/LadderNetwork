import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

from ladder_network import tensor_utils


def gaussian_combinator(decoder,
                        encoder,
                        scope=None):
    with tf.name_scope(scope, 'gaussian_combinator', [decoder, encoder]):
        n_hidden_units = tensor_utils.get_shape(decoder)[-1]

        # combinator weights
        a1 = slim.variable('a1',
                           initializer=tf.zeros_initializer,
                           shape=[n_hidden_units],
                           trainable=True)

        a2 = slim.variable('a2',
                           initializer=tf.constant_initializer(1.0),
                           shape=[n_hidden_units],
                           trainable=True)

        a3 = slim.variable('a3',
                           initializer=tf.zeros_initializer,
                           shape=[n_hidden_units],
                           trainable=True)

        a4 = slim.variable('a4',
                           initializer=tf.zeros_initializer,
                           shape=[n_hidden_units],
                           trainable=True)

        a5 = slim.variable('a5',
                           initializer=tf.zeros_initializer,
                           shape=[n_hidden_units],
                           trainable=True)

        a6 = slim.variable('a6',
                           initializer=tf.zeros_initializer,
                           shape=[n_hidden_units],
                           trainable=True)

        a7 = slim.variable('a7',
                           initializer=tf.constant_initializer(1.0),
                           shape=[n_hidden_units],
                           trainable=True)

        a8 = slim.variable('a8',
                           initializer=tf.zeros_initializer,
                           shape=[n_hidden_units],
                           trainable=True)

        a9 = slim.variable('a9',
                           initializer=tf.zeros_initializer,
                           shape=[n_hidden_units],
                           trainable=True)

        a10 = slim.variable('a10',
                           initializer=tf.zeros_initializer,
                           shape=[n_hidden_units],
                           trainable=True)

        mu = a1 * tf.sigmoid(a2 * decoder + a3) + a4 * decoder + a5
        v = a6 * tf.sigmoid(a7 * decoder + a3) + a9 * decoder + a10

        z_est = (encoder - mu) * v + mu

        return z_est
