import time
import sys
import logging
import importlib
import numpy as np
import tensorflow as tf
import os


from utils.adapt_data import batch_fill
from utils.evaluations import save_results
import dagmm.gmm_utils as gmm

RANDOM_SEED = 13
FREQ_PRINT = 5000 # print frequency image tensorboard [20]
METHOD = "inception"
def display_parameters(batch_size, starting_lr,
                       l1, l2, label):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('Weights loss - l1:', l1, '; l2:', l2)
    print('Anomalous label: ', label)

def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(dataset, K, l1, l2, label, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "dagmm/train_logs/{}/K{}/{}_{}/{}/{}".format(dataset, K, l1, l2,
                                                               label, rd)

def reconstruction_error(x, x_rec):
    return tf.norm(x-x_rec, axis=1)

def train_and_test(dataset, nb_epochs, K, l1, l2, label,
                   random_seed):

    """ Runs the DAGMM on the specified dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("DAGMM.train.{}.{}".format(dataset,label))

    # Import model and data
    model = importlib.import_module('dagmm.{}_utilities'.format(dataset))
    data = importlib.import_module("data.{}".format(dataset))

    # Parameters
    starting_lr = model.params["learning_rate"]
    batch_size = model.params["batch_size"]
    if l1==-1: l1 = model.params["l1"]
    if l2==-1: l2 = model.params["l2"]
    if K==-1: K = model.params["K"]

    # Placeholders

    x_pl = tf.placeholder(tf.float32, data.get_shape_input())
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    logger.info('Building training graph...')

    logger.warning("The DAGMM is training with the following parameters:")
    display_parameters(batch_size, starting_lr, l1, l2,
                       label)


    global_step = tf.Variable(0, name='global_step', trainable=False)

    enc = model.encoder
    dec = model.decoder
    feat_ex = model.feature_extractor
    est = model.estimator

    #feature extraction for images
    if model.params["is_image"] and not METHOD=="pca":
         x_features = image_features.extract_features(x_pl)
    else:
         x_features = x_pl
    n_features = x_features.shape[1]

    with tf.variable_scope('encoder_model'):
        z_c = enc(x_features, is_training=is_training_pl)  
    
    with tf.variable_scope('decoder_model'):
        x_rec = dec(z_c, n_features, is_training=is_training_pl)

    with tf.variable_scope('feature_extractor_model'):
        x_flat = tf.layers.flatten(x_features)
        x_rec_flat = tf.layers.flatten(x_rec)
        z_r = feat_ex(x_flat, x_rec_flat)

    z = tf.concat([z_c, z_r], axis=1)

    with tf.variable_scope('estimator_model'):
        gamma = est(z, K, is_training=is_training_pl)

    with tf.variable_scope('gmm'):
        energy, penalty = gmm.compute_energy_and_penalty(z, gamma, is_training_pl)

    with tf.name_scope('loss_functions'):
        # reconstruction error
        rec_error = reconstruction_error(x_flat, x_rec_flat)
        loss_rec = tf.reduce_mean(rec_error)

        # probabilities to observe
        loss_energy = tf.reduce_mean(energy)
 
        # full loss
        full_loss = loss_rec + l1*loss_energy + l2*penalty



    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.5, name='dis_optimizer')

        train_op = optimizer.minimize(full_loss, global_step=global_step)

        with tf.name_scope('predictions'):
            # Highest 20% are anomalous
            if dataset=="kdd":
                per = tf.contrib.distributions.percentile(energy, 80)
            else:
                per = tf.contrib.distributions.percentile(energy, 80)
            y_pred = tf.greater_equal(energy, per)
           
    with tf.name_scope('summary'):
        with tf.name_scope('loss_summary'):
            tf.summary.scalar('loss_rec', loss_rec, ['loss'])
            tf.summary.scalar('mean_energy', loss_energy, ['loss'])
            tf.summary.scalar('penalty', penalty, ['loss'])
            tf.summary.scalar('full_loss', full_loss, ['loss'])

        sum_op_loss = tf.summary.merge_all('loss')

    # Data
    logger.info('Data loading...')

    trainx, trainy = data.get_train(label)
    trainx_copy = trainx.copy()
    testx, testy = data.get_test(label)
    
    if model.params["is_image"] and METHOD=="pca":
       logger.info('PCA...')
       trainx = trainx.reshape([trainx.shape[0], -1])
       testx = testx.reshape([testx.shape[0], -1])
       trainx, testx = image_features.pca(trainx, testx, 20)
       logger.info('Done')

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)
    

    logdir = create_logdir(dataset, K, l1, l2, label, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=10)

    logger.info('Start training...')
    with sv.managed_session() as sess:

        logger.info('Initialization done')
        writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

             # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
            train_loss_rec, train_loss = [0, 0]

            # training
            for t in range(nr_batches_train):

                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = ran_from + batch_size
                feed_dict = {x_pl: trainx[ran_from:ran_to], 
                             is_training_pl: True,
                             learning_rate:lr}

                _, lrec, loss, sm, step = sess.run([train_op,
                                              loss_rec,
                                              full_loss,
                                              sum_op_loss,
                                              global_step],
                                              feed_dict=feed_dict)
                train_loss_rec += lrec
                train_loss += loss
                writer.add_summary(sm, step)#train_batch)

                if np.isnan(loss):
                    logger.info("Loss is nan - Stopping")
                    break

                train_batch += 1
       
            if np.isnan(loss):
                logger.info("Loss is nan - Stopping")
                break

            train_loss_rec /= nr_batches_train
            train_loss /= nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss rec = %.4f "
                  "| loss = %.4f"
                  % (epoch, time.time() - begin, train_loss_rec,
                     train_loss))

            epoch += 1


        logger.warning('Testing evaluation...')
         
        inds = rng.permutation(testx.shape[0])
    
        ##TESTING PER BATCHS
        inference_time = [] 
        scores = []
        for t in range(nr_batches_test+1):
            ran_from = t * batch_size
            ran_to = min(ran_from + batch_size, testx.shape[0])
            feed_dict = {x_pl: testx[ran_from:ran_to], 
                         is_training_pl: False}
            begin_val = time.time()
            if l1>0:
                scoresb, step = sess.run([energy, global_step], feed_dict=feed_dict)
            else:
                scoresb, step = sess.run([rec_error, global_step], feed_dict=feed_dict)
            scores.append(scoresb)
            inference_time.append(time.time() - begin_val)
        scores = np.concatenate(scores, axis = 0)

        logger.warning('Testing : inference time is %.4f' % (
            np.mean(inference_time)))

        #scores = np.array(scores)
        save_results(scores, testy, 'dagmm/K{}'.format(K), dataset, None, str(l1)+"_"+str(l2), label,
                     random_seed, step)
 

def run(args):
    """ Runs the training process"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(args.rd)
        train_and_test(args.dataset, args.nb_epochs, args.K, args.l1, args.l2,
                       args.label, args.rd)
