import numpy as np

def batch_fill(testx, batch_size):
    """ Quick and dirty hack for filling smaller batch

    :param testx:
    :param batch_size:
    :return:
    """
    nr_batches_test = int(testx.shape[0] / batch_size)
    ran_from = nr_batches_test * batch_size
    ran_to = (nr_batches_test + 1) * batch_size
    size = testx[ran_from:ran_to].shape[0]
    new_shape = [batch_size - size]+list(testx.shape[1:])
    fill = np.ones(new_shape)
    return np.concatenate([testx[ran_from:ran_to], fill], axis=0), size

def adapt_labels_outlier_task(true_labels, label):
    """Adapt labels to anomaly detection context

    Args :
            true_labels (list): list of ints
            label (int): label which is considered inlier
    Returns :
            true_labels (list): list of labels, 1 for anomalous and 0 for normal
    """
    if label == 1:
        (true_labels[true_labels == label], true_labels[true_labels != label]) = (1, 0)
        true_labels = [1] * true_labels.shape[0] - true_labels
    else:
        (true_labels[true_labels != label], true_labels[true_labels == label]) = (1, 0)
    return true_labels





