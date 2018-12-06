import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import os
from utils.adapt_data import batch_fill
from utils.evaluations import save_results, heatmap
from utils.constants import IMAGES_DATASETS

FREQ_PRINT = 200 # print frequency image tensorboard [20]
FREQ_EV = 1
PATIENCE = 20
STEPS_NUMBER = 500

def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay,
                       weight, degree, label):
    """See parameters
    """
    print('Batch size: ', batch_size)       
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Degree for L norms: ', degree)
    print('Anomalous label: ', label)

def display_progression_epoch(j, id_max):
    """See epoch progression
    """
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(dataset, weight, label, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "train_logs/{}/anogan/label{}/weight{}/rd{}".format(
        dataset, label, weight, rd)

def train_and_test(dataset, nb_epochs, weight, degree, random_seed, label,
                   enable_sm, score_method, enable_early_stop):

    """ Runs the AnoGAN on the specified dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        dataset (str): name of the dataset
        nb_epochs (int): number of epochs
        weight (float): weight in the inverting loss function
        degree (int): degree of the norm in the feature matching
        random_seed (int): trying different seeds for averaging the results
        label (int): label which is normal for image experiments
        enable_sm (bool): allow TF summaries for monitoring the training
        score_method (str): which metric to use for the ablation study
        enable_early_stop (bool): allow early stopping for determining the number of epochs
    """
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    logger = logging.getLogger("AnoGAN.run.{}.{}".format(dataset, label))

    # Import model and data
    network = importlib.import_module('anogan.{}_utilities'.format(dataset))
    data = importlib.import_module("data.{}".format(dataset))

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.999

    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Placeholders
    x_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    logger.info('Data loading...')
    trainx, trainy = data.get_train(label)
    if enable_early_stop: 
        validx, validy = data.get_valid(label)
        nr_batches_valid = int(validx.shape[0] / batch_size)
    trainx_copy = trainx.copy()
    testx, testy = data.get_test(label)

    rng = np.random.RandomState(random_seed)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    logger.info('Building graph...')
    logger.warn("The GAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, degree, label)

    gen = network.generator
    dis = network.discriminator

    # Sample noise from random normal distribution
    random_z = tf.random_normal([batch_size, latent_dim], mean=0.0,
                                stddev=1.0, name='random_z')
    # Generate images with generator
    x_gen = gen(random_z, is_training=is_training_pl)

    real_d, inter_layer_real = dis(x_pl, is_training=is_training_pl)
    fake_d, inter_layer_fake = dis(x_gen, is_training=is_training_pl,
                                   reuse=True)

    with tf.name_scope('loss_functions'):
        # Calculate seperate losses for discriminator with real and fake images
        real_discriminator_loss = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(real_d), real_d,
            scope='real_discriminator_loss')
        fake_discriminator_loss = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(fake_d), fake_d,
            scope='fake_discriminator_loss')
        # Add discriminator losses
        loss_discriminator = real_discriminator_loss + fake_discriminator_loss
        # Calculate loss for generator by flipping label on discriminator output
        loss_generator = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(fake_d), fake_d, scope='generator_loss')

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope='discriminator')
        gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope='generator')

        update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                           scope='generator')
        update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                           scope='discriminator')

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                  beta1=0.5)

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer.minimize(loss_generator, var_list=gvars,
                                            global_step=global_step)

        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for inference
        def train_op_with_ema_dependency(vars, op):
            ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            maintain_averages_op = ema.apply(vars)
            with tf.control_dependencies([op]):
                train_op = tf.group(maintain_averages_op)
            return train_op, ema

        train_gen_op, gen_ema = train_op_with_ema_dependency(gvars, gen_op)
        train_dis_op, dis_ema = train_op_with_ema_dependency(dvars, dis_op)

    ### Testing ###
    with tf.variable_scope("latent_variable"):
        z_optim = tf.get_variable(name='z_optim',
                                  shape= [batch_size, latent_dim],
                                  initializer=tf.truncated_normal_initializer())
        reinit_z = z_optim.initializer

    # EMA
    x_gen_ema = gen(random_z, is_training=is_training_pl,
                    getter=get_getter(gen_ema), reuse=True)
    rec_x_ema = gen(z_optim, is_training=is_training_pl,
                        getter=get_getter(gen_ema), reuse=True)
    # Pass real and fake images into discriminator separately
    real_d_ema, inter_layer_real_ema = dis(x_pl,
                                           is_training=is_training_pl,
                                           getter=get_getter(gen_ema),
                                           reuse=True)
    fake_d_ema, inter_layer_fake_ema = dis(rec_x_ema,
                                           is_training=is_training_pl,
                                           getter=get_getter(gen_ema),
                                           reuse=True)

    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = x_pl - rec_x_ema
            delta_flat = tf.contrib.layers.flatten(delta)
            reconstruction_score = tf.norm(delta_flat, ord=degree, axis=1,
                              keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_scores'):

            if score_method == 'cross-e':
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(fake_d_ema),logits=fake_d_ema)

            else:
                fm = inter_layer_real_ema - inter_layer_fake_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1,
                                     keep_dims=False, name='d_loss')

            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            loss_invert = weight * reconstruction_score \
                                  + (1 - weight) * dis_score

    rec_error_valid = tf.reduce_mean(loss_invert)

    with tf.variable_scope("Test_learning_rate"):
        step_lr = tf.Variable(0, trainable=False)
        learning_rate_invert = 0.001
        reinit_lr = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope="Test_learning_rate"))

    with tf.name_scope('Test_optimizer'):
        invert_op = tf.train.AdamOptimizer(learning_rate_invert).\
            minimize(loss_invert,global_step=step_lr, var_list=[z_optim],
                     name='optimizer')
        reinit_optim = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope='Test_optimizer'))

    reinit_test_graph_op = [reinit_z, reinit_lr, reinit_optim]

    with tf.name_scope("Scores"):
        list_scores = loss_invert

    if enable_sm:
        with tf.name_scope('training_summary'):
            with tf.name_scope('dis_summary'):
                tf.summary.scalar('real_discriminator_loss',
                                  real_discriminator_loss, ['dis'])
                tf.summary.scalar('fake_discriminator_loss',
                                  fake_discriminator_loss, ['dis'])
                tf.summary.scalar('discriminator_loss', loss_discriminator, ['dis'])

            with tf.name_scope('gen_summary'):
                tf.summary.scalar('loss_generator', loss_generator, ['gen'])

            with tf.name_scope('img_summary'):
                heatmap_pl_latent = tf.placeholder(tf.float32,
                                                   shape=(1, 480, 640, 3),
                                                   name="heatmap_pl_latent")
                sum_op_latent = tf.summary.image('heatmap_latent',
                                                 heatmap_pl_latent)

            with tf.name_scope('validation_summary'):
                tf.summary.scalar('valid', rec_error_valid, ['v'])

            if dataset in IMAGES_DATASETS:
                with tf.name_scope('image_summary'):
                    tf.summary.image('reconstruct', x_gen, 8, ['image'])
                    tf.summary.image('input_images', x_pl, 8, ['image'])

            else:
                heatmap_pl_rec = tf.placeholder(tf.float32, shape=(1, 480, 640, 3),
                                                name="heatmap_pl_rec")
                with tf.name_scope('image_summary'):
                    tf.summary.image('heatmap_rec', heatmap_pl_rec, 1, ['image'])

            sum_op_dis = tf.summary.merge_all('dis')
            sum_op_gen = tf.summary.merge_all('gen')
            sum_op = tf.summary.merge([sum_op_dis, sum_op_gen])
            sum_op_im = tf.summary.merge_all('image')
            sum_op_valid = tf.summary.merge_all('v')

    logdir = create_logdir(dataset, weight, label, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=None)

    logger.info('Start training...')
    with sv.managed_session(config=config) as sess:

        logger.info('Initialization done')

        writer = tf.summary.FileWriter(logdir, sess.graph)

        train_batch = 0
        epoch = 0
        best_valid_loss = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling unl dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

            train_loss_dis, train_loss_gen = [0, 0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)

                # construct randomly permuted minibatches
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {x_pl: trainx[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, ld, step = sess.run([train_dis_op, loss_discriminator, global_step], feed_dict=feed_dict)
                train_loss_dis += ld

                # train generator
                feed_dict = {x_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, lg = sess.run([train_gen_op, loss_generator], feed_dict=feed_dict)
                train_loss_gen += lg

                if enable_sm:
                    sm = sess.run(sum_op, feed_dict=feed_dict)
                    writer.add_summary(sm, step)

                    if t % FREQ_PRINT == 0:  # inspect reconstruction
                        # t = np.random.randint(0,400)
                        # ran_from = t
                        # ran_to = t + batch_size
                        # sm = sess.run(sum_op_im, feed_dict={x_pl: trainx[ran_from:ran_to],is_training_pl: False})
                        # writer.add_summary(sm, train_batch)

                        # data = sess.run(z_gen, feed_dict={
                        #     x_pl: trainx[ran_from:ran_to],
                        #     is_training_pl: False})
                        # data = np.expand_dims(heatmap(data), axis=0)
                        # sml = sess.run(sum_op_latent, feed_dict={
                        #     heatmap_pl_latent: data,
                        #     is_training_pl: False})
                        #
                        # writer.add_summary(sml, train_batch)

                        if dataset in IMAGES_DATASETS:
                            sm = sess.run(sum_op_im, feed_dict={
                                x_pl: trainx[ran_from:ran_to],
                                is_training_pl: False})
                        #
                        # else:
                        #     data = sess.run(z_gen, feed_dict={
                        #         x_pl: trainx[ran_from:ran_to],
                        #         z_pl: np.random.normal(
                        #             size=[batch_size, latent_dim]),
                        #         is_training_pl: False})
                        #     data = np.expand_dims(heatmap(data), axis=0)
                        #     sm = sess.run(sum_op_im, feed_dict={
                        #         heatmap_pl_rec: data,
                        #         is_training_pl: False})
                            writer.add_summary(sm, step)#train_batch)
                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_dis /= nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_dis))

            ##EARLY STOPPING
            if (epoch + 1) % FREQ_EV == 0 and enable_early_stop:
                logger.info('Validation...')

                inds = rng.permutation(validx.shape[0])
                validx = validx[inds]  # shuffling  dataset
                validy = validy[inds]  # shuffling  dataset

                valid_loss = 0

                # Create scores
                for t in range(nr_batches_valid):
                    # construct randomly permuted minibatches
                    display_progression_epoch(t, nr_batches_valid)
                    ran_from = t * batch_size
                    ran_to = (t + 1) * batch_size

                    feed_dict = {x_pl: validx[ran_from:ran_to],
                                 is_training_pl: False}
                    for _ in range(STEPS_NUMBER):
                        _ = sess.run(invert_op, feed_dict=feed_dict)
                    vl = sess.run(rec_error_valid,
                                       feed_dict=feed_dict)
                    valid_loss += vl
                    sess.run(reinit_test_graph_op)

                valid_loss /= nr_batches_valid
                sess.run(reinit_test_graph_op)

                if enable_sm:
                    sm = sess.run(sum_op_valid, feed_dict=feed_dict)
                    writer.add_summary(sm, step)  # train_batch)

                logger.info(
                    'Validation: valid loss {:.4f}'.format(valid_loss))

                if valid_loss < best_valid_loss or epoch == FREQ_EV - 1:

                    best_valid_loss = valid_loss
                    logger.info("Best model - valid loss = {:.4f} - saving...".format(
                            best_valid_loss))
                    sv.saver.save(sess, logdir + '/model.ckpt',
                                  global_step=step)
                    nb_without_improvements = 0
                else:
                    nb_without_improvements += FREQ_EV

                if nb_without_improvements > PATIENCE:
                    sv.request_stop()
                    logger.warning(
                        "Early stopping at epoch {} with weights from epoch {}".format(
                            epoch, epoch - nb_without_improvements))

            epoch += 1

        logger.warn('Testing evaluation...')
        step = sess.run(global_step)
        sv.saver.save(sess, logdir + '/model.ckpt', global_step=step)

        rect_x, rec_error, latent, scores = [], [], [], []
        inference_time = []
        
        # Create scores
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            display_progression_epoch(t, nr_batches_test)
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            feed_dict = {x_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

            for _ in range(STEPS_NUMBER):
                _ = sess.run(invert_op, feed_dict=feed_dict)

            brect_x, brec_error, bscores, blatent = sess.run([rec_x_ema, reconstruction_score, loss_invert, z_optim], feed_dict=feed_dict)
            rect_x.append(brect_x)
            rec_error.append(brec_error)
            scores.append(bscores)
            latent.append(blatent)
            sess.run(reinit_test_graph_op)

            inference_time.append(time.time() - begin_val_batch)

        logger.info('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))

        if testx.shape[0] % batch_size != 0:
            batch, size = batch_fill(testx, batch_size)
            feed_dict = {x_pl: batch,
                         is_training_pl: False}
            for _ in range(STEPS_NUMBER):
                _ = sess.run(invert_op, feed_dict=feed_dict)
            brect_x, brec_error, bscores, blatent = sess.run([rec_x_ema, reconstruction_score, loss_invert, z_optim], feed_dict=feed_dict)
            rect_x.append(brect_x[:size])
            rec_error.append(brec_error[:size])
            scores.append(bscores[:size])
            latent.append(blatent[:size])
            sess.run(reinit_test_graph_op)

        rect_x = np.concatenate(rect_x, axis=0)
        rec_error = np.concatenate(rec_error, axis=0)
        scores = np.concatenate(scores, axis=0)
        latent = np.concatenate(latent, axis=0)
        save_results(scores, testy, 'anogan', dataset, score_method,
                     weight, label, random_seed)

def run(args):
    """ Runs the training process"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(args.rd)
        train_and_test(args.dataset, args.nb_epochs, args.w, args.d, args.rd, args.label,
                      args.enable_sm, args.m, args.enable_early_stop)
