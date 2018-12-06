import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import os
from utils.adapt_data import batch_fill
from utils.evaluations import save_grid_plot, save_results
from utils.constants import IMAGES_DATASETS

RANDOM_SEED = 13
FREQ_PRINT = 50 # print frequency image tensorboard [20]
FREQ_EV = 1
STRIP_EV = 5
FREQ_SNAP = 1000
BATCH_EV = 50
ENABLE_EV = False
PATIENCE = 25

def display_parameters(batch_size, starting_lr, label):
    """See parameters
    """
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('Anomalous label: ', label)

def display_progression_epoch(j, id_max):
    """See epoch progression
    """
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(dataset, label, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "dsebm/train_logs/{}/label{}/rd{}".format(
        dataset,label, rd)


def train_and_test(dataset, nb_epochs, random_seed, label):

    """ Runs DSEBM on available datasets 

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        dataset (string): dataset to run the model on 
        nb_epochs (int): number of epochs
        random_seed (int): trying different seeds for averaging the results
        label (int): label which is normal for image experiments
        anomaly_type (string): "novelty" for 100% normal samples in the training set
                               "outlier" for a contamined training set 
        anomaly_proportion (float): if "outlier", anomaly proportion in the training set
    """
    logger = logging.getLogger("DSEBM.run.{}.{}".format(
        dataset, label))

    # Import model and data
    network = importlib.import_module('dsebm.{}_utilities'.format(dataset))
    data = importlib.import_module("data.{}".format(dataset))

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size

    # Placeholders
    x_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(),
                          name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    #test
    y_true = tf.placeholder(tf.int32, shape=[None], name="y_true")

    logger.info('Building training graph...')
    logger.warn("The DSEBM is training with the following parameters:")
    display_parameters(batch_size, starting_lr, label)

    net = network.network

    global_step = tf.train.get_or_create_global_step()

    noise = tf.random_normal(shape=tf.shape(x_pl), mean=0.0, stddev=1.,
                             dtype=tf.float32)
    x_noise = x_pl + noise

    with tf.variable_scope('network'):
        b_prime_shape = list(data.get_shape_input())
        b_prime_shape[0] = batch_size
        b_prime = tf.get_variable(name='b_prime', shape=b_prime_shape)#tf.shape(x_pl))
        net_out = net(x_pl, is_training=is_training_pl)
        net_out_noise = net(x_noise, is_training=is_training_pl, reuse=True)

    with tf.name_scope('energies'):
        energy = 0.5 * tf.reduce_sum(tf.square(x_pl - b_prime)) \
                 - tf.reduce_sum(net_out)
        
        energy_noise = 0.5 * tf.reduce_sum(tf.square(x_noise - b_prime)) \
                       - tf.reduce_sum(net_out_noise)

    with tf.name_scope('reconstructions'):
        # reconstruction
        grad = tf.gradients(energy, x_pl)
        fx = x_pl - tf.gradients(energy, x_pl)
        fx = tf.squeeze(fx, axis=0)
        fx_noise = x_noise - tf.gradients(energy_noise, x_noise)

    with tf.name_scope("loss_function"):
        # DSEBM for images
        if len(data.get_shape_input())==4:
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_pl - fx_noise),
                                                axis=[1,2,3]))
        # DSEBM for tabular data        
        else:
            loss = tf.reduce_mean(tf.square(x_pl - fx_noise))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        netvars = [var for var in tvars if 'network' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_net = [x for x in update_ops if ('network' in x.name)]

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                  name='optimizer')

        with tf.control_dependencies(update_ops_net):
            train_op = optimizer.minimize(loss, var_list=netvars, global_step=global_step)

    with tf.variable_scope('Scores'):

        with tf.name_scope('Energy_score'):
            flat = tf.layers.flatten(x_pl - b_prime)
            if len(data.get_shape_input())==4:
                list_scores_energy = 0.5 * tf.reduce_sum(tf.square(flat), axis=1) \
                                           - tf.reduce_sum(net_out, axis=[1, 2, 3])
            else:
                list_scores_energy = 0.5 * tf.reduce_sum(tf.square(flat), axis=1) \
                           - tf.reduce_sum(net_out, axis=1)
        with tf.name_scope('Reconstruction_score'):
            delta = x_pl - fx
            delta_flat = tf.layers.flatten(delta)
            list_scores_reconstruction = tf.norm(delta_flat, ord=2, axis=1,
                                                keepdims=False,
                                                name='reconstruction')

     #   with tf.name_scope('predictions'):
     #       # Highest 20% are anomalous
     #       if dataset=="kdd":
     #           per = tf.contrib.distributions.percentile(list_scores_energy, 80)
     #       else:
     #           per = tf.contrib.distributions.percentile(list_scores_energy, 95)
     #       y_pred = tf.greater_equal(list_scores_energy, per)
     #      
     #       #y_test_true = tf.cast(y_test_true, tf.float32)
     #       cm = tf.confusion_matrix(y_true, y_pred, num_classes=2)
     #       recall = cm[1,1]/(cm[1,0]+cm[1,1])
     #       precision = cm[1,1]/(cm[0,1]+cm[1,1])
     #       f1 = 2*precision*recall/(precision + recall)
    
    with tf.name_scope('training_summary'):

        tf.summary.scalar('score_matching_loss', loss, ['net'])
        tf.summary.scalar('energy', energy, ['net'])

        if dataset in IMAGES_DATASETS:
            with tf.name_scope('image_summary'):
                tf.summary.image('reconstruct', fx, 6, ['image'])
                tf.summary.image('input_images', x_pl, 6, ['image'])
                sum_op_im = tf.summary.merge_all('image')

        sum_op_net = tf.summary.merge_all('net')

    logdir = create_logdir(dataset, label, random_seed)
    
    sv = tf.train.Supervisor(logdir=logdir+"/train", save_summaries_secs=None,
                             save_model_secs=None)
    
    # Data
    logger.info('Data loading...')
    trainx, trainy = data.get_train(label)
    trainx_copy = trainx.copy()
    if dataset in IMAGES_DATASETS: validx, validy = data.get_valid(label)
    testx, testy = data.get_test(label)

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    if dataset in IMAGES_DATASETS: nr_batches_valid = int(validx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)
    
    logger.info("Train: {} samples in {} batches".format(trainx.shape[0], nr_batches_train))
    if dataset in IMAGES_DATASETS: logger.info("Valid: {} samples in {} batches".format(validx.shape[0], nr_batches_valid))
    logger.info("Test:  {} samples in {} batches".format(testx.shape[0], nr_batches_test))

    logger.info('Start training...')
    with sv.managed_session() as sess:
        logger.info('Initialization done')
        
        train_writer = tf.summary.FileWriter(logdir+"/train", sess.graph)
        valid_writer = tf.summary.FileWriter(logdir+"/valid", sess.graph)

        train_batch = 0
        epoch = 0
        best_valid_loss = 0
        train_losses = [0]*STRIP_EV

        while not sv.should_stop() and epoch < nb_epochs:
            lr = starting_lr

            begin = time.time()
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling unl dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

            losses, energies = [0, 0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)

                # construct randomly permuted minibatches
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train the net
                feed_dict = {x_pl: trainx[ran_from:ran_to],
                             is_training_pl:True,
                             learning_rate:lr}
                _, ld, en, sm, step = sess.run([train_op, loss, energy, sum_op_net, global_step], feed_dict=feed_dict)
                losses += ld
                energies += en
                train_writer.add_summary(sm, step)

                if t % FREQ_PRINT == 0 and dataset in IMAGES_DATASETS:  # inspect reconstruction
                    t= np.random.randint(0,40)
                    ran_from = t
                    ran_to = t + batch_size
                    sm = sess.run(sum_op_im, feed_dict={x_pl: trainx[ran_from:ran_to],is_training_pl: False})
                    train_writer.add_summary(sm, step)

                train_batch += 1

            losses /= nr_batches_train
            energies /= nr_batches_train
            # Remembering loss for early stopping
            train_losses[epoch%STRIP_EV] = losses

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss = %.4f | energy = %.4f "
                  % (epoch, time.time() - begin, losses, energies))

            if (epoch + 1) % FREQ_SNAP == 0 and dataset in IMAGES_DATASETS:

                print("Take a snap of the reconstructions...")
                x = trainx[:batch_size]
                feed_dict = {x_pl: x,
                             is_training_pl: False}

                rect_x = sess.run(fx, feed_dict=feed_dict)
                nama_e_wa = "dsebm/reconstructions/{}/{}/" \
                            "{}_epoch{}".format(dataset,
                                                label,
                                                random_seed, epoch)
                nb_imgs = 50
                save_grid_plot(x[:nb_imgs], rect_x[:nb_imgs], nama_e_wa, nb_imgs)

            if (epoch + 1) % FREQ_EV == 0 and dataset in IMAGES_DATASETS:
                logger.info("Validation")
                inds = rng.permutation(validx.shape[0])
                validx = validx[inds] # shuffling  dataset
                validy = validy[inds] # shuffling  dataset
                valid_loss = 0
                for t in range(nr_batches_valid):
                    display_progression_epoch(t, nr_batches_valid)

                    # construct randomly permuted minibatches
                    ran_from = t * batch_size
                    ran_to = (t + 1) * batch_size

                    # train the net
                    feed_dict = {x_pl: validx[ran_from:ran_to],
                                 y_true: validy[ran_from:ran_to],
                                 is_training_pl:False}

                    vl, sm, step = sess.run([loss, sum_op_net, global_step], feed_dict=feed_dict)
                    valid_writer.add_summary(sm, step+t)#train_batch)
                    valid_loss += vl 
               
                valid_loss /= nr_batches_valid 
                


                # train the net
                
                logger.info("Validation loss at step "+str(step)+":"+str(valid_loss))
                ##EARLY STOPPING
                #UPDATE WEIGHTS
                if valid_loss<best_valid_loss or epoch==FREQ_EV-1:
                    best_valid_loss = valid_loss
                    logger.info("Best model - loss={} - saving...".format(best_valid_loss))
                    sv.saver.save(sess, logdir+'/train/model.ckpt', 
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
        scores_e = []
        scores_r = []
        inference_time = []

        # Create scores
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            feed_dict = {x_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

            scores_e += sess.run(list_scores_energy,
                                 feed_dict=feed_dict).tolist()

            scores_r += sess.run(list_scores_reconstruction,
                                 feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_val_batch)

        logger.info('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))

        if testx.shape[0] % batch_size != 0:
            batch, size = batch_fill(testx, batch_size)
            feed_dict = {x_pl: batch,
                         is_training_pl: False}
            batch_score_e = sess.run(list_scores_energy,
                                   feed_dict=feed_dict).tolist()
            batch_score_r = sess.run(list_scores_reconstruction,
                                   feed_dict=feed_dict).tolist()
            scores_e += batch_score_e[:size]
            scores_r += batch_score_r[:size]

        save_results(scores_e, testy, 'dsebm', dataset, 'energy', "test", label,
                     random_seed, step)
        save_results(scores_r, testy, 'dsebm', dataset, 'reconstruction', "test",
                      label, random_seed, step)

def run(args):
    """ Runs the training process"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(args.rd)
        train_and_test(args.dataset, args.nb_epochs, args.rd, args.label)
