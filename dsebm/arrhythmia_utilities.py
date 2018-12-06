# TensorFlow implementation of a DCGAN model for arrhythmia

import tensorflow as tf

learning_rate = 0.0001
batch_size = 26
init_kernel = tf.contrib.layers.xavier_initializer()

def UnPooling2x2ZeroFilled(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
        return ret


def network(x_inp, is_training=False, getter=None, reuse=False):
    """ Network architecture in tensorflow

    Discriminates between real data and generated data

    Note:
        Provides histogram and distribution tensorflow summaries

    Args:
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.variable_scope('network', reuse=reuse, custom_getter=getter):
       kernel_dense = tf.get_variable('kernel_dense', [274, 10], initializer=init_kernel)
       bias_dense = tf.get_variable('bias_dense', [10])
       kernel_dense2 = tf.get_variable('kernel_dense2', [10, 2], initializer=init_kernel)
       bias_dense2 = tf.get_variable('bias_dense2', [2])
       bias_inv_dense2 = tf.get_variable('bias_inv_dense2', [10])
       bias_inv_dense = tf.get_variable('bias_inv_dense', [274])


       x = tf.nn.softplus(tf.matmul(x_inp, kernel_dense) + bias_dense)
 
       x = tf.nn.softplus(tf.matmul(x, kernel_dense2) + bias_dense2)
       
   ###INVERSE LAYERS 

       x = tf.nn.softplus(tf.matmul(x, tf.transpose(kernel_dense2)) + bias_inv_dense2)
       
       x = tf.nn.softplus(tf.matmul(x, tf.transpose(kernel_dense)) + bias_inv_dense)

    return x


