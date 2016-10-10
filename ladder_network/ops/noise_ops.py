import tensorflow as tf
import tensorflow.contrib.slim as slim

from ladder_network.ops import training

@slim.add_arg_scope
def dropout(tensor_in, keep_prob=0.5, scope=None):
    with tf.name_scope(scope, 'dropout', [tensor_in]):
        tensor_in = tf.convert_to_tensor(tensor_in)
        tensor_with_dropout = tf.nn.dropout(tensor_in, keep_prob)

        return training.in_train_phase(tensor_with_dropout, tensor_in)


@slim.add_arg_scope
def bernoulli_choice(neg_val, pos_val, shape, proba=0.5, seed=123, scope=None):
    with tf.name_scope(scope, 'bernoulli_choice', []):
        proba = tf.convert_to_tensor(proba,
                                     dtype=tf.float32,
                                     name='proba')

        neg_val = tf.convert_to_tensor(neg_val,
                                       dtype=tf.float32,
                                       name='neg_val')
        pos_val = tf.convert_to_tensor(pos_val,
                                       dtype=tf.float32,
                                       name='pos_val')

        random_tensor = tf.random_uniform(shape, seed=seed)
        return tf.select(random_tensor < proba,
                         tf.ones_like(random_tensor) * pos_val,
                         tf.ones_like(random_tensor) * neg_val)


@slim.add_arg_scope
def gaussian_noise(tensor_in, stddev=1.0, random_state=42, scope=None):
    with tf.name_scope(scope, 'gaussian_noise', [tensor_in]) as sc:
        tensor_in = tf.convert_to_tensor(tensor_in)
        input_shape = tf.shape(tensor_in)
        noisy_tensor_in = tensor_in + tf.random_normal(input_shape,
                                                       mean=0,
                                                       stddev=stddev,
                                                       seed=random_state,
                                                       dtype=tensor_in.dtype)

        return training.in_train_phase(noisy_tensor_in, tensor_in)


@slim.add_arg_scope
def salt_and_pepper_noise(tensor_in,
                          min_val=0,
                          max_val=1,
                          corruption_proba=0.5,
                          random_state=42,
                          scope=None):
    with tf.variable_scope(scope, 'salt_and_pepper_noise', [tensor_in]):
        tensor_in = tf.convert_to_tensor(tensor_in)

        # corruption mask
        corruption_proba = tf.convert_to_tensor(corruption_proba,
                                                name='corruption_proba')

        random_tensor = corruption_proba
        random_tensor += tf.random_uniform(tf.shape(tensor_in),
                                           seed=random_state,
                                           dtype=tensor_in.dtype)
        # 0 if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob]
        binary_tensor = tf.cast(tf.floor(random_tensor), tf.bool)


        # bernoulli salt & pepper noise
        salt_and_pepper = bernoulli_choice(neg_val=min_val,
                                           pos_val=max_val,
                                           shape=tf.shape(tensor_in))

        noisy_tensor_in = tf.select(binary_tensor,
                                    salt_and_pepper,
                                    tensor_in,
                                    name='noisy_tensor_in')

        return training.in_train_phase(noisy_tensor_in, tensor_in)


@slim.add_arg_scope
def noise_layer(tensor_in,
                noise_type=None,
                corruption_proba=0.5,
                min_val=0,
                max_val=1,
                gaussian_stddev=1.0,
                random_state=42,
                scope=None):
    """Apply noise to the input Tensor. Currently supported noise functions
    are dropout noise (set certain features to zero) or additive gaussian noise.

    Parameters
    ----------
    tensor_in : tf.Tensor
        The input tensor.

    noise_type : str, {'dropout', 'gaussian', 'salt_and_pepper'}
        The type of noise to use for the data denoising step. Can either
        apply dropout to features, additive gaussian noise, or
        salt and pepper noise. If None then no noise is applied.

    corruption_proba : float (default=0.5)
        Dropout probability. If None or noise_type == 'gaussian', then
        no dropout is used.

    min_val : float (default=0)
        Minimum value for salt and pepper noise. Should be the minimum
        value in the dataset.

    max_val : float (default=1)
        Maximum value for salt and pepper noise. Should be the maximum
        value in the dataset.

    gaussian_stddev : float (default=1.0)
        Standard deviation of the additive gaussian noise. If None or
        noise_type == 'dropout', then no gaussian noise is added.

    random_state : int
        Seed for the random number generator.

    name : str
        Name of op scope.
    """
    with tf.name_scope(scope, 'noise_layer', [tensor_in]):
        tensor_in = tf.convert_to_tensor(tensor_in)

        if corruption_proba and noise_type == 'dropout':
            noise_tensor_in = dropout(tensor_in,
                                      keep_prob=1-corruption_proba,
                                      random_state=random_state)
        elif gaussian_stddev and noise_type == 'gaussian':
            noise_tensor_in = gaussian_noise(tensor_in,
                                             stddev=gaussian_stddev,
                                             random_state=random_state)
        elif (min_val is not None and max_val is not None and
                noise_type == 'salt_and_pepper'):
            noise_tensor_in = salt_and_pepper_noise(
                    tensor_in,
                    min_val=min_val,
                    max_val=max_val,
                    corruption_proba=corruption_proba,
                    random_state=random_state)
        else:  # no_op
            noise_tensor_in = tensor_in

        return noise_tensor_in
