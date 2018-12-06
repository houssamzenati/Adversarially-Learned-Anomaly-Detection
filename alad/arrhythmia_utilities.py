"""

Arrhythmia ALAD architecture.

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

def encoder(x_inp, is_training=False, getter=None, reuse=False,
            do_spectral_norm=False):
    """ Encoder architecture in tensorflow

    Maps the data into the latent space

    Args:
        x_inp (tensor): input data for the encoder.
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not

    Returns:
        net (tensor): last activation layer of the encoder

    """
    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                                  units=256,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=latent_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')
    return net

def decoder(z_inp, is_training=False, getter=None, reuse=False):
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

def discriminator_xz(x_inp, z_inp, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=False):
    """ Discriminator architecture in tensorflow

    Discriminates between pairs (E(x), x) and (z, G(z))

    Args:
        x_inp (tensor): input data for the discriminator.
        z_inp (tensor): input variable in the latent space
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.variable_scope('discriminator_xz', reuse=reuse, custom_getter=getter):
        # D(x)
        name_x = 'x_layer_1'
        with tf.variable_scope(name_x):
            x = tf.layers.dense(x_inp,
                          units=128,
                          kernel_initializer=init_kernel,
                          name='fc')
            x = tf.layers.batch_normalization(x,
                                                training=is_training,
                                                name='batch_normalization')
            x = leakyReLu(x)

        # D(z)
        name_z = 'z_layer_1'
        with tf.variable_scope(name_z):
            z = tf.layers.dense(z_inp, 128, kernel_initializer=init_kernel)
            z = leakyReLu(z)
            z = tf.layers.dropout(z, rate=0.5, name='dropout', training=is_training)


        # D(x,z)
        y = tf.concat([x, z], axis=1)

        name_y = 'y_layer_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(y,
                                256,
                                kernel_initializer=init_kernel)
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.5, name='dropout', training=is_training)


        intermediate_layer = y

        name_y = 'y_layer_2'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     1,
                                     kernel_initializer=init_kernel)

    return logits, intermediate_layer

def discriminator_xx(x, rec_x, is_training=False,getter=None, reuse=False,
            do_spectral_norm=False):
    """ Discriminator architecture in tensorflow

    Discriminates between (x,x) and (x,rec_x)

    Args:
        x (tensor): input from the data space
        rec_x (tensor): reconstructed data
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.variable_scope('discriminator_xx', reuse=reuse, custom_getter=getter):
        net = tf.concat([x, rec_x], axis=1)

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                              units=256,
                              kernel_initializer=init_kernel,
                              name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net, rate=0.2, name='dropout', training=is_training)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                              units=128,
                              kernel_initializer=init_kernel,
                              name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net, rate=0.2, name='dropout', training=is_training)

        intermediate_layer = net

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            logits = tf.layers.dense(net,
                              units=1,
                              kernel_initializer=init_kernel,
                              name='fc')

    return logits, intermediate_layer

def discriminator_zz(z, rec_z, is_training=False, getter=None, reuse=False,
            do_spectral_norm=False):
    """ Discriminator architecture in tensorflow

    Discriminates between (z,z) and (z,rec_z)

    Args:
        z (tensor): input from the latent space
        rec_z (tensor): reconstructed data
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.variable_scope('discriminator_zz', reuse=reuse, custom_getter=getter):

        net = tf.concat([z, rec_z], axis=-1)
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                   units=64,
                                   kernel_initializer=init_kernel,
                                   name='fc')

            net = leakyReLu(net, 0.2, name='conv1/leaky_relu')
            net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                      training=is_training)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                   units=32,
                                   kernel_initializer=init_kernel,
                                   name='fc')

            net = leakyReLu(net, 0.2, name='conv1/leaky_relu')
            net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                training=is_training)
        intermediate_layer = net

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            logits = tf.layers.dense(net,
                                   units=1,
                                   kernel_initializer=init_kernel,
                                   name='fc')

    return logits, intermediate_layer