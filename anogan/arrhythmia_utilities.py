"""

Arrhythmia GAN architecture.

Generator (decoder), encoder and discriminator.

"""
import tensorflow as tf
from utils import sn

learning_rate = 1e-5
batch_size = 32
latent_dim = 64
init_kernel = tf.contrib.layers.xavier_initializer()

def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
    else:
        return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

def generator(z_inp, is_training=False, getter=None, reuse=False):
    """ Generator architecture in tensorflow

    Generates data from the latent space

    Args:
        z_inp (tensor): input variable in the latent space
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not

    Returns:
        net (tensor): last activation layer of the generator

    """
    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=256,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=274,
                                  kernel_initializer=init_kernel,
                                  name='fc')

    return net

def discriminator(x_inp, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=False):
    """ Discriminator architecture in tensorflow

    Discriminates between pairs (E(x), x) and (z, G(z))

    Args:
        x_inp (tensor): input data for the discriminator.
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                          units=256,
                          kernel_initializer=init_kernel,
                          name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net, rate=0.2, name='dropout', training=is_training)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                128,
                                kernel_initializer=init_kernel)
            net = leakyReLu(net)
            net = tf.layers.dropout(net, rate=0.5, name='dropout', training=is_training)


        intermediate_layer = net

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            logits = tf.layers.dense(net,
                                     1,
                                     kernel_initializer=init_kernel)

    return logits, intermediate_layer
