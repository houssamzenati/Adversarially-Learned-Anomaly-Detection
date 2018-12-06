# TensorFlow implementation of a DSEBM model for CIFAR10

import tensorflow as tf

init_kernel = tf.contrib.layers.xavier_initializer()
image_size = 32

learning_rate = 0.003
batch_size = 32
kernel_conv_size = 3
filters_conv = 64
filters_fc = 128
strides_conv = 2

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
       kernel_conv = tf.get_variable('kernel_conv', [kernel_conv_size, kernel_conv_size, 3, filters_conv], initializer=init_kernel)
       conv_output_size = int(image_size*image_size/4/strides_conv/strides_conv*filters_conv)
       kernel_dense = tf.get_variable('kernel_dense', [conv_output_size, filters_fc], initializer=init_kernel)
       bias_dense = tf.get_variable('bias_dense', [filters_fc])
       bias_inv_dense = tf.get_variable('bias_inv_dense', [conv_output_size])

       x = tf.nn.conv2d(x_inp,
                        kernel_conv, 
                        [1, strides_conv, strides_conv, 1], 
                        'SAME')
       x = tf.nn.softplus(x)
       x = tf.nn.pool(x, (2, 2), "MAX", "SAME", strides=(2, 2))

       x = tf.contrib.layers.flatten(x)
       x = tf.nn.softplus(tf.matmul(x, kernel_dense) + bias_dense)
       
    ###INVERSE LAYERS

       x = tf.nn.softplus(tf.matmul(x, tf.transpose(kernel_dense)) + bias_inv_dense)
       new_image_size = int(image_size/2/strides_conv)
       x = tf.reshape(x, [-1, new_image_size, new_image_size, filters_conv])
       
       x = UnPooling2x2ZeroFilled(x)
       x = tf.nn.conv2d_transpose(x,
                                  kernel_conv,
                                  tf.shape(x_inp), 
                                  [1, strides_conv, strides_conv, 1],
                                  'SAME')

       x = tf.nn.softplus(x, name='softplus')


    return x


