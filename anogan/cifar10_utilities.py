"""

CIFAR10 DCGAN architecture.

Generator and discriminator.

"""
import tensorflow as tf

learning_rate = 0.0002
batch_size = 32
latent_dim = 100
init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.01)

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
        net = tf.reshape(z_inp, [-1, 1, 1, latent_dim])

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                             filters=512,
                                             kernel_size=4,
                                             strides=2,
                                             padding='VALID',
                                             kernel_initializer=init_kernel,
                                             name='tconv1')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv1/batch_normalization')

            net = tf.nn.relu(net, name='tconv1/relu')

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                             filters=256,
                                             kernel_size=4,
                                             strides=2,
                                             padding='SAME',
                                             kernel_initializer=init_kernel,
                                             name='tconv2')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv2/batch_normalization')
            net = tf.nn.relu(net, name='tconv2/relu')

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                             filters=128,
                                             kernel_size=4,
                                             strides=2,
                                             padding='SAME',
                                             kernel_initializer=init_kernel,
                                             name='tconv3')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv3/batch_normalization')
            net = tf.nn.relu(net, name='tconv3/relu')

        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                             filters=3,
                                             kernel_size=4,
                                             strides=2,
                                             padding='SAME',
                                             kernel_initializer=init_kernel,
                                             name='tconv4')

            net = tf.tanh(net, name='tconv4/tanh')

    return net

def discriminator(x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow

    Discriminates between real data and generated data

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
            net = tf.layers.conv2d(x_inp,
                          filters=128,
                          kernel_size=4,
                          strides=2,
                          padding='SAME',
                          kernel_initializer=init_kernel,
                          name='conv1')

            net = leakyReLu(net, 0.2, name='conv1/leaky_relu')

        name_net = 'x_layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d(net,
                          filters=256,
                          kernel_size=4,
                          strides=2,
                          padding='SAME',
                          kernel_initializer=init_kernel,
                          name='conv2')

            net = tf.layers.batch_normalization(net,
                                              training=is_training,
                                              name='conv2/batch_normalization')

            net = leakyReLu(net, 0.2, name='conv2/leaky_relu')

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d(net,
                          filters=512,
                          kernel_size=4,
                          strides=2,
                          padding='SAME',
                          kernel_initializer=init_kernel,
                          name='conv3')

            net = tf.layers.batch_normalization(net,
                                              training=is_training,
                                              name='conv3/batch_normalization')

            net = leakyReLu(net, 0.2, name='conv3/leaky_relu')

        net = tf.contrib.layers.flatten(net)

        intermediate_layer = net

        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=1,
                                  kernel_initializer=init_kernel,
                                  name='fc')

            logits = tf.squeeze(net)

    return logits, intermediate_layer