#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:07:15 2018

@author: Manon Romain
"""
import numpy as np
import tensorflow as tf
import math as m

tfd = tf.contrib.distributions

def tricky_divide(x, y):
    return tf.transpose(tf.transpose(x)/y)

def tricky_multiply(x, y):
    return tf.transpose(tf.transpose(x)*y)


def compute_energy_and_penalty(z, gammas, is_training):
    """
    Computes the energy and penalty of the GMM as described
    in Zong, Bo et al. “Deep Autoencoding Gaussian Mixture Model
                        for Unsupervised Anomaly Detection.” (2018).

    K: number of mixtures in the GMM
    N: number of samples
    M: number of features of z

    Args:
        - z :       [N, M] - the reconstruction features concatenated
                    with the latent representation of x

        - gamma :   [N, K] - density of probability - output of the
                    estimation network

    Returns:

        - E(z):     Tensor of shape [batch_size] - energy of the
                    GMM computed by the networks

        - P(sigma): Scalar - Penalty appled to small values in diagonal entries of
                    covariance matrices

    Note: could probably be simplified

    """
    #shapes
    K = gammas.get_shape()[1]
    M = z.get_shape()[1]
    with tf.variable_scope('gmm_parameters'):
        phis = tf.get_variable('phis', shape=[K], initializer=tf.ones_initializer(), dtype=tf.float32, trainable=False)
        mus = tf.get_variable('mus', shape=[K, M], initializer=tf.ones_initializer(), dtype=tf.float32, trainable=False)

        init_sigmas = 0.5 * np.expand_dims(np.identity(M), axis=0) 
        init_sigmas = np.tile(init_sigmas, [2, 1, 1])
        init_sigmas = tf.constant_initializer(init_sigmas)
        sigmas = tf.get_variable('sigmas', shape=[K, M, M], initializer=init_sigmas, dtype=tf.float32, trainable=False)

        sums = tf.reduce_sum(gammas, axis=0)
        sums_exp_dims = tf.expand_dims(sums, axis=-1)

        phis_ = tf.reduce_mean(gammas, axis=0)
        mus_ = tf.matmul(gammas, z, transpose_a=True) / sums_exp_dims

        def assign_training_phis_mus():
            with tf.control_dependencies([phis.assign(phis_), mus.assign(mus_)]):
                return [tf.identity(phis), tf.identity(mus)]

        phis, mus = tf.cond(is_training, assign_training_phis_mus, lambda: [phis, mus])
        phis_exp_dims = tf.expand_dims(phis, axis=0)
        phis_exp_dims = tf.expand_dims(phis_exp_dims, axis=-1)
        phis_exp_dims = tf.expand_dims(phis_exp_dims, axis=-1)

        zs_exp_dims = tf.expand_dims(z, 1)
        zs_exp_dims = tf.expand_dims(zs_exp_dims, -1)
        mus_exp_dims = tf.expand_dims(mus, 0)
        mus_exp_dims = tf.expand_dims(mus_exp_dims, -1)

        zs_minus_mus = zs_exp_dims - mus_exp_dims

        sigmas_ = tf.matmul(zs_minus_mus, zs_minus_mus, transpose_b=True)
        broadcast_gammas = tf.expand_dims(gammas, axis=-1)
        broadcast_gammas = tf.expand_dims(broadcast_gammas, axis=-1)
        sigmas_ = broadcast_gammas * sigmas_
        sigmas_ = tf.reduce_sum(sigmas_, axis=0)
        sigmas_ = sigmas_ / tf.expand_dims(sums_exp_dims, axis=-1)
        sigmas_ = add_noise(sigmas_)

        def assign_training_sigmas():
            with tf.control_dependencies([sigmas.assign(sigmas_)]):
                return tf.identity(sigmas)

        sigmas = tf.cond(is_training, assign_training_sigmas, lambda: sigmas)

    inversed_sigmas = tf.expand_dims(tf.matrix_inverse(sigmas), axis=0)
    inversed_sigmas = tf.tile(inversed_sigmas, [tf.shape(zs_minus_mus)[0], 1, 1, 1])
    energy = tf.matmul(zs_minus_mus, inversed_sigmas, transpose_a=True)
    energy = tf.matmul(energy, zs_minus_mus)
    energy = tf.squeeze(phis_exp_dims * tf.exp(-0.5 * energy), axis=[2, 3])
    energy_divided_by = tf.expand_dims(tf.sqrt(2.0 * m.pi * tf.matrix_determinant(sigmas)), axis=0) + 1e-12
    energy = tf.reduce_sum(energy / energy_divided_by, axis=1) + 1e-12
    energy = -1.0 * tf.log(energy)

    penalty = 1.0 / tf.matrix_diag_part(sigmas)
    penalty = tf.reduce_sum(penalty)
    return energy, penalty

def add_noise(mat, stdev=0.001):
    """
    :param mat: should be of shape(k, d, d)
    :param stdev: the standard deviation of noise
    :return: a matrix with little noises
    """
    with tf.name_scope('gaussian_noise'):
        dims = mat.get_shape().as_list()[1]
        noise = stdev + tf.random_normal([dims], 0, stdev * 1e-1)
        noise = tf.diag(noise)
        noise = tf.expand_dims(noise, axis=0)
        noise = tf.tile(noise, (mat.get_shape()[0], 1, 1))
    return mat + noise

