import sys
import os
from six.moves import urllib
from scipy.io import loadmat
import logging
from utils.adapt_data import adapt_labels_outlier_task
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_train(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("train", label, centered, normalize)

def get_test(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("test", label, centered, normalize)
    
def get_valid(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("valid", label, centered, normalize)

def get_shape_input():
    return (None, 32, 32, 3)

def get_shape_input_flatten():
    return (None, 3072)

def get_anomalous_proportion():
    return 0.9

def _get_adapted_dataset(split, label, centered, normalize):
    """
    Gets the adapted dataset for the experiments
    Args :
            split (str): train or test
            mode (str): inlier or outlier
            label (int): int in range 0 to 10, is the class/digit
                         which is considered inlier or outlier
            centered (bool): (Default=False) data centered to [-1, 1]
    Returns :
            (tuple): <training, testing> images and labels
            :type label: object
    """
    dataset = {}
    dataset['x_train'],  dataset['y_train'] = load(subset='train')
    dataset['x_test'], dataset['y_test'] = load(subset='test')

    def rescale(mat):
        return np.transpose(mat, (3, 0, 1, 2))

    dataset['x_train'] = rescale(dataset['x_train'])
    dataset['x_test'] = rescale(dataset['x_test'])

    if normalize:
        dataset['x_train'] = dataset['x_train'].astype(np.float32) / 255.0
        dataset['x_test'] = dataset['x_test'].astype(np.float32) / 255.0
    if centered:
        dataset['x_train'] = dataset['x_train'].astype(np.float32) * 2. - 1.
        dataset['x_test'] = dataset['x_test'].astype(np.float32) * 2. - 1.

    full_x_data = np.concatenate([dataset['x_train'], dataset['x_test']], axis=0)
    full_y_data = np.concatenate([dataset['y_train'], dataset['y_test']], axis=0)

    #
    dataset['x_train'], dataset['x_test'], \
    dataset['y_train'], dataset['y_test'] = train_test_split(full_x_data,
                                                             full_y_data,
                                                             test_size=0.2,
                                                             random_state=42)
    
    dataset['x_train'], dataset['x_valid'], \
    dataset['y_train'], dataset['y_valid'] = train_test_split(dataset['x_train'],
                                                             dataset['y_train'],
                                                             test_size=0.25,
                                                             random_state=42)

    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if label != -1:

        if split in ['train', 'valid']:

            inliers = dataset[key_img][dataset[key_lbl] == label], \
                      dataset[key_lbl][dataset[key_lbl] == label]
            outliers = dataset[key_img][dataset[key_lbl] != label], \
                       dataset[key_lbl][dataset[key_lbl] != label]

            dataset[key_img], dataset[key_lbl] = inliers

            dataset[key_lbl] = adapt_labels_outlier_task(dataset[key_lbl],
                                                         label)
        else:
            dataset[key_lbl] = adapt_labels_outlier_task(dataset[key_lbl],
                                                         label)

    return (dataset[key_img], dataset[key_lbl])

def maybe_download(data_dir):
    new_data_dir = os.path.join(data_dir, 'svhn')
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', new_data_dir+'/train_32x32.mat', _progress)
        filepath, _ = urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', new_data_dir+'/test_32x32.mat', _progress)

def load(data_dir='./data/svhn', subset='train'):
    maybe_download(data_dir)
    if subset=='train':
        train_data = loadmat(os.path.join(data_dir, 'svhn') + '/train_32x32.mat')
        trainx = train_data['X']
        trainy = train_data['y'].flatten()
        trainy[trainy==10] = 0
        return trainx, trainy
    elif subset=='test':
        test_data = loadmat(os.path.join(data_dir, 'svhn') + '/test_32x32.mat')
        testx = test_data['X']
        testy = test_data['y'].flatten()
        testy[testy==10] = 0
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')
