import numpy as np
import tensorflow as tf


def get_shape(tensor):
    """Get the shape of a tensor as a python list."""
    if isinstance(tensor, tf.Tensor):
        return tensor.get_shape().as_list()
    elif isinstance(tensor, (np.array, list, tuple)):
        return np.shape(tensor)


def to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def get_fans(shape):
    """determine the input and output nodes going into a layer.
    Convolutional layers must take into account the receptive field size.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        # tensorflow kernel shape: (kernel_w, kernel_h, input_channels, output_channesl)
        n1, n2 = shape[2:]
        receptive_field_size = np.prod(shape[:2])
        fan_in = n1 * receptive_field_size
        fan_out = n2 * receptive_field_size
    return fan_in, fan_out
