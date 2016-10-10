"""
Training Ops
------------

Because some ops have different behavior at training and testing time
(such as dropout or batch normalization), we implement a boolean variable
`is_training` that indicates whether the network is used for training or inference.
This variable is stored in the tf.collection `is_training`, and is the unique
element of it.

Two operations to update that variable (set it to True or False),
are stored in another tf.collection `is_training_ops` with 2 elements:
[set_training_mode_op, set_predicting_mode_op]. So invoking the first element
will enable training mode, while the second one will enable predicting mode.
"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def is_training(is_training=False, session=None):
    """is_training.

    Set the graph training mode. This is meant to be used to control ops
    that have different output at training and testing time, such as
    dropout and batch normalization.

    Parameters
    ----------
    is_training : bool
        What the training mode of the current graph should be set too.
    session : tf.Session (optional)
        The session that owns the current computational graph.

    Examples
    --------
    ...
    >>> training_mode = skflow.get_training_mode()
    >>> my_conditional_op = tf.cond(training_mode, if_yes_op, if_no_op)
    >>> skflow.is_training(True)
    >>> session.run(my_conditional_op)
    if_yes_op
    >>> skflow.is_training(False)
    >>> session.run(my_conditional_op)
    if_no_op
    ...

    Returns
    -------
    A `bool`, True if training, False if inference.
    """
    if not session:
        session = tf.get_default_session()

    with session.graph.as_default():
        init_training_mode()

        if is_training:
            tf.get_collection('is_training_ops')[0].eval(session=session)
        else:
            tf.get_collection('is_training_ops')[1].eval(session=session)


def get_training_mode():
    """get_training_mode

    Returns variable in-use to set training mode.

    Returns
    -------
    A `tf.Variable`, the training mode holder.
    """
    init_training_mode()
    coll = tf.get_collection('is_training')
    return coll[0]


def init_training_mode():
    """init_training_mode

    Creates `is_training` variable and its ops if they haven't been created yet.
    """
    coll = tf.get_collection('is_training')
    if len(coll) == 0:
        training_phase = tf.get_variable(
            'is_training', dtype=tf.bool, shape=[],
            initializer=tf.constant_initializer(False),
            trainable=False)

        tf.add_to_collection('is_training', training_phase)

        set_training = tf.assign(training_phase, True)
        set_inference = tf.assign(training_phase, False)

        tf.add_to_collection('is_training_ops', set_training)
        tf.add_to_collection('is_training_ops', set_inference)

def in_train_phase(x, alt):
    is_training = get_training_mode()
    return control_flow_ops.cond(
        is_training,
        lambda: x,
        lambda: alt)
