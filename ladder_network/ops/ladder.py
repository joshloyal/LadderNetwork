from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import tensorflow.contrib.losses as losses

from ladder_network import tensor_utils
from ladder_network import initializers
from ladder_network import activations
from ladder_network.ops import noise_ops
from ladder_network.ops import batch_norm_ops
from ladder_network.ops import combinator_ops
from ladder_network.ops import training_ops

# Define output collections
CLEAN_ENCODER_OUTPUTS = 'clean_encoder_outputs'
CLEAN_ENCODER_HIDDEN_STATE = 'clean_encoder_hidden_state'
NOISY_ENCODER_OUTPUTS = 'noisy_encoder_outputs'
NOISY_ENCODER_HIDDEN_STATE = 'noisy_encoder_hidden_state'


def lookup_encoder_outputs(encoder_id, encoder_type='clean'):
    if encoder_type == 'clean':
        collection = CLEAN_ENCODER_OUTPUTS
    else:
        collection = NOISY_ENCODER_OUTPUTS

    # I'm unsure if order is preserved in collections, so for now
    # let's assume it's not...
    for encoder in tf.get_collection(collection):
        if encoder.alias == '{}_output_{:d}'.format(encoder_type, encoder_id):
            return encoder

    raise LookupError('Could not find outputs for '
                      'encoder_id: %s' % encoder_id)


def lookup_encoder_hidden_state(encoder_id, encoder_type='clean'):
    if encoder_type == 'clean':
        collection = CLEAN_ENCODER_HIDDEN_STATE
    else:
        collection = NOISY_ENCODER_HIDDEN_STATE

    # I'm unsure if order is preserved in collections, so for now
    # let's assume it's not...
    for encoder in tf.get_collection(collection):
        if encoder.alias == '{}_hidden_{:d}'.format(encoder_type, encoder_id):
            return encoder

    raise LookupError('Could not find hidden state for '
                      'encoder_id: %s' % encoder_id)


def lookup_encoder_batch_statistics(encoder_id, encoder_type='clean'):
    if encoder_type == 'clean':
        collection = CLEAN_ENCODER_OUTPUTS + '/batch_norm_ops'
    else:
        collection = NOISY_ENCODER_OUTPUTS + '/batch_norm_ops'

    mean, variance = None, None
    for batch_op in tf.get_collection(collection):
        if 'encoding_layer_{:d}'.format(encoder_id) in batch_op.alias:
            if 'mean' in batch_op.alias:
                mean = batch_op
            elif 'variance' in batch_op.alias:
                variance = batch_op

    if mean is None or variance is None:
        raise LookupError('Could not find batch statistics for '
                          'encoder_id: %s' % encoder_id)

    return mean, variance


@slim.add_arg_scope
def fully_connected_encoder(tensor_in,
                   n_hidden_units,
                   activation='relu',
                   noise_type=None,
                   gaussian_stddev=1.0,
                   random_state=42,
                   reuse=None,
                   hidden_outputs_collections=None,
                   outputs_collections=None,
                   scope=None):
    with tf.variable_scope(scope, 'fully_connected_encoder',
                           [tensor_in]) as scope:
        tensor_in = tf.convert_to_tensor(tensor_in)

        # dense encoding layer (re-use)
        encoder = layers.fully_connected(
                tensor_in,
                num_outputs=n_hidden_units,
                activation_fn=activations.get(activation),
                weights_initializer=initializers.get('glorot_uniform'),
                reuse=reuse,
                scope='weight_layer')

        # this is a noisy encoder
        if noise_type:
            # noisy encoder batch statistics
            encoder = batch_norm_ops.batch_normalization(
                    encoder,
                    outputs_collections=outputs_collections + '/batch_norm_ops',
                    scope=scope.name + '/noisy_batch_norm')

            encoder = noise_ops.noise_layer(encoder,
                                            noise_type='gaussian',
                                            gaussian_stddev=gaussian_stddev,
                                            random_state=random_state)
        else:
            # clean encoder batch statistics
            encoder = batch_norm_ops.batch_normalization(
                    encoder,
                    outputs_collections=outputs_collections + '/batch_norm_ops',
                    scope=scope.name + 'clean_batch_norm')

        # add hidden states to collection for latter access
        if noise_type:
            layers.utils.collect_named_outputs(
                    hidden_outputs_collections,
                    'noisy_hidden_{:s}'.format(scope.name[-1]),
                    encoder)
        else:
            layers.utils.collect_named_outputs(
                    hidden_outputs_collections,
                    'clean_hidden_{:s}'.format(scope.name[-1]),
                    encoder)

        # add the gamma and beta scaling from batchnorm
        encoder = batch_norm_ops.scale_and_center(encoder,
                                                  reuse=reuse,
                                                  scope='scale_and_center')

        # Need to deal with softmax here to!
        outputs = activations.get(activation)(encoder)

        if noise_type:
            return layers.utils.collect_named_outputs(
                outputs_collections,
                'noisy_output_{:s}'.format(scope.name[-1]),
                outputs)
        else:
            return layers.utils.collect_named_outputs(
                    outputs_collections,
                    'clean_output_{:s}'.format(scope.name[-1]),
                    outputs)


@slim.add_arg_scope
def fully_connected_decoder(tensor_in,
                            noisy_encoder,
                            activation='relu',
                            linear_transformation=False,
                            random_state=42,
                            scope=None):

    with tf.variable_scope(scope, 'fully_connected_decoder',
                           [tensor_in, noisy_encoder]) as scope:
        if linear_transformation:
            n_hidden_units = tensor_utils.get_shape(noisy_encoder)[-1]
            decoder = layers.fully_connected(
                    tensor_in,
                    num_outputs=n_hidden_units,
                    activation_fn=activations.get(activation),
                    weights_initializer=initializers.get('glorot_uniform'),
                    scope='weight_layer')
        else:
            decoder = tensor_in

        decoder = batch_norm_ops.batch_normalization(decoder,
                                                     scope='decoder_batch_norm')

        decoder = combinator_ops.gaussian_combinator(decoder, noisy_encoder)

        return decoder


@slim.add_arg_scope
def ladder_network(X, y,
                   hidden_units,
                   activation='relu',
                   gaussian_stddev=1.0,
                   random_state=42,
                   scope=None):
    """Fully supervised ladder network."""
    with tf.variable_scope(scope, 'ladder_network', [X, y]) as scope:
        X = tf.convert_to_tensor(X)
        y = tf.convert_to_tensor(y)

        n_layers = len(hidden_units)
        n_classes = tensor_utils.get_shape(y)[-1]

        noisy_input = noise_ops.noise_layer(X,
                                            noise_type='gaussian',
                                            gaussian_stddev=gaussian_stddev,
                                            random_state=random_state)

        # First run the corrupt data through the corrupted encoder
        encoder = noisy_input
        for layer_number, n_hidden_units in enumerate(hidden_units):
            encoder = fully_connected_encoder(encoder,
                                     n_hidden_units,
                                     activation=activation,
                                     noise_type='gaussian',
                                     gaussian_stddev=gaussian_stddev,
                                     random_state=random_state,
                                     hidden_outputs_collections=NOISY_ENCODER_HIDDEN_STATE,
                                     outputs_collections=NOISY_ENCODER_OUTPUTS,
                                     scope='encoding_layer_{:d}'.format(
                                         layer_number))

        # noisy classification head
        noisy_logits = fully_connected_encoder(encoder,
                                 n_hidden_units=n_classes,
                                 activation='identity',
                                 noise_type='gaussian',
                                 gaussian_stddev=gaussian_stddev,
                                 random_state=random_state,
                                 hidden_outputs_collections=NOISY_ENCODER_HIDDEN_STATE,
                                 outputs_collections=NOISY_ENCODER_OUTPUTS,
                                 scope='encoding_layer_{:d}'.format(n_layers))
        supervised_loss = losses.softmax_cross_entropy(noisy_logits, y)
        noisy_preds = tf.nn.softmax(noisy_logits)

        # Now run the clean data through the clean encoder
        encoder = X
        for layer_number, n_hidden_units in enumerate(hidden_units):
            encoder = fully_connected_encoder(encoder,
                                     n_hidden_units,
                                     activation=activation,
                                     noise_type=None,
                                     random_state=random_state,
                                     reuse=True,  # re-use variables from previous encoder
                                     hidden_outputs_collections=CLEAN_ENCODER_HIDDEN_STATE,
                                     outputs_collections=CLEAN_ENCODER_OUTPUTS,
                                     scope='encoding_layer_{:d}'.format(
                                         layer_number))

        # clean classification head
        clean_logits = fully_connected_encoder(encoder,
                                 n_hidden_units=n_classes,
                                 activation='identity',
                                 noise_type=None,
                                 random_state=random_state,
                                 reuse=True,
                                 hidden_outputs_collections=CLEAN_ENCODER_HIDDEN_STATE,
                                 outputs_collections=CLEAN_ENCODER_OUTPUTS,
                                 scope='encoding_layer_{:d}'.format(n_layers))
        predictions = tf.nn.softmax(clean_logits)

        # go back through the decoder
        reconstruction_costs = []
        decoder = noisy_preds
        for layer_number in xrange(n_layers, -1, -1):
            hidden_encoder = lookup_encoder_hidden_state(layer_number, encoder_type='noisy')
            use_linear_transform = False if layer_number == n_layers else True
            decoder = fully_connected_decoder(decoder,
                                     hidden_encoder,
                                     linear_transformation=use_linear_transform,
                                     scope='decoding_layer_{:d}'.format(
                                         layer_number))

            # calculate reconstruction loss
            encoder_mean, encoder_variance = lookup_encoder_batch_statistics(
                    layer_number, encoder_type='clean')
            z_est_bn = (decoder - encoder_mean) / encoder_variance

            # calculate unsupervised loss
            clean_hidden_encoder = lookup_encoder_hidden_state(
                layer_number, encoder_type='clean')
            loss = tf.reduce_sum((z_est_bn - clean_hidden_encoder)**2, 1)
            unsupervised_loss = tf.reduce_mean(loss)
            reconstruction_costs.append(unsupervised_loss)

        total_reconstruction_loss = tf.add_n(reconstruction_costs)
        total_loss = supervised_loss + 0.10 * total_reconstruction_loss

        return predictions, total_loss


if __name__ == '__main__':
    weight_shape = (100, 20)
    X = slim.variable('X',
                      shape=weight_shape,
                      initializer=initializers.get('glorot_uniform'))
    y_shape = (100, 10)
    y = slim.variable('y',
                      shape=y_shape,
                      initializer=tf.constant_initializer(1.))
    predictions, loss = ladder_network(X, y, hidden_units=[128, 32, 16])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_ops.is_training(True, session=sess)
        for i in range(10):
            sess.run([predictions, loss])
