### Credits https://github.com/taki0112/Spectral_Normalization-Tensorflow

import tensorflow as tf

def conv2d(inputs, filters, kernel_size, strides=1, padding='valid',
           use_bias=True, kernel_initializer=None,
           bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
           name=None,reuse=None):

    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, inputs.get_shape()[-1], filters], initializer=kernel_initializer,
                            regularizer=kernel_regularizer)
        bias = tf.get_variable("bias", [filters], initializer=bias_initializer)
        x = tf.nn.conv2d(input=inputs, filter=spectral_norm(w),
                         strides=[1, strides, strides, 1], padding=padding)
        if use_bias :
            x = tf.nn.bias_add(x, bias)

    return x

def dense(inputs, units, use_bias=True, kernel_initializer=None,
          bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
          name=None,reuse=None):

    with tf.variable_scope(name, reuse=reuse):
        inputs = tf.contrib.layers.flatten(inputs)
        shape = inputs.get_shape().as_list()
        channels = shape[-1]

        w = tf.get_variable("kernel", [channels, units], tf.float32,
                                 initializer=kernel_initializer, regularizer=kernel_regularizer)
        if use_bias :
            bias = tf.get_variable("bias", [units],
                                   initializer=bias_initializer)

            x = tf.matmul(inputs, spectral_norm(w)) + bias
        else :
            x = tf.matmul(inputs, spectral_norm(w))

    return x

def spectral_norm(w, iteration=1, eps=1e-12):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + eps)

        u_ = tf.matmul(v_hat, w)
        u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + eps)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
