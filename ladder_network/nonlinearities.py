import numpy as np
import tensorflow as tf

from ladder_network.wrappers import Registry

activations = Registry('Activation')


@activations.register('relu')
def relu(x, name=None):
    return tf.nn.relu(x, name=name)


@activations.register('paired_relu')
def paired_relu(x, name='paired_relu'):
    """Paired ReLUs.

    Normal ReLUs suffer from the 'dying ReLU' problem, which occurs when
    large gradients activates the weights in such a way that unit can never
    activate on any datapoint again (gradients are zero for x < 0). On averge
    up to 40% on ReLU neurons can be dead in a network. The Paired Relu
    is similar to the CReLU proposed by Shang et. al except here the sign
    of the output is preserved.

    The Paired ReLU calculates both max(x, 0) and min(x, 0) on the input
    so that there is always some path for gradients to flow. In this
    case the ouput tensor size is doubled and each path has its own weights.
    """
    relu_pairs = [tf.nn.relu(x), -tf.nn.relu(-x)]
    return skflow.ops.merge(relu_pairs, mode='concat')


@activations.register('leaky_relu')
def leaky_relu(x, alpha=0., max_value=None, name=''):
    """Leaky Rectified Linear Unit (ReLU)

    Parameters
    ----------
    x : Tensor
    alpha : float
        Slope of negative section.
    max_value : float
        Saturation threshold
    name : str
        A name for this activation op
    """
    op_scope = skflow.tensor.get_scope(x)
    with tf.name_scope(op_scope + name) as scope:
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if max_value is not None:
            x = tf.clip_by_value(
                x, tf.cast(0., dtype=tf.float32), tf.cast(max_value, dtype=tf.float32))
        if isinstance(alpha, (tuple, list, np.ndarray)) or np.isscalar(alpha):
            alpha = tf.constant(alpha, dtype=tf.float32)
        x -= alpha * negative_part
        return x


@activations.register('prelu')
def prelu(x, alphas_init=0.25, name='prelu'):
    """PReLU.

    Parameteric Rectified Linear Unit

    Parameters
    ----------
    x : Tensor
    alphas_init : float
        Value to initialize coefficients
        (the default is 0.25, which is used in the original paper).
    name : str
        Name for the op scope.

    References
    ----------
    .. [1] He, et al.
       "Delving Deep into Rectifiers: Surpassing Human-Level Performance on
       ImageNet Classification."
       <http://arxiv.org/pdf/1502.01852v1.pdf>
    """
    a_shape = skflow.tensor.get_shape(x)[1:]
    op_scope = skflow.tensor.get_scope(x)
    with tf.variable_op_scope([x], op_scope + name, 'prelu') as scope:
        a_init = tf.constant_initializer(alphas_init)
        alphas = skflow.tensor.variable('alphas',
                                        shape=a_shape,
                                        initializer=a_init)
        x = tf.nn.relu(x) + tf.mul(alphas, (x - tf.abs(x))) * 0.5

    # save the alphas in the tensor to make it easy to grab later
    x.alphas = alphas

    return x


@activations.register('elu')
def elu(x, name=None):
    return tf.nn.elu(x, name=name)


@activations.register('sigmoid')
def sigmoid(x, name=None):
    return tf.nn.sigmoid(x, name=name)


@activations.register('tanh')
def tanh(x, name=None):
    return tf.nn.tanh(x, name=name)


@activations.register('softplus')
def softplus(x, name=None):
    return tf.nn.softplus(x, name=name)

@activations.register('identity')
def identity(x, name=None):
    return tf.identity(x, name=name)

@activations.register('softmax')
def softmax(x, name=None):
    return tf.nn.softmax(x, name=name)
