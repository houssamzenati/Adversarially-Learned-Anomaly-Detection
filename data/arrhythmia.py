import logging
import numpy as np
import pandas as pd 
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def get_train(label=0, scale=False, *args):
    """Get training dataset for Thyroid dataset"""
    return _get_adapted_dataset("train", scale)

def get_test(label=0, scale=False, *args):
    """Get testing dataset for Thyroid dataset"""
    return _get_adapted_dataset("test", scale)

def get_valid(label=0, scale=False, *args):
    """Get validation dataset for Thyroid dataset"""
    return None

def get_shape_input():
    """Get shape of the dataset for Thyroid dataset"""
    return (None, 274)

def get_shape_input_flatten():
    """Get shape of the dataset for Thyroid dataset"""
    return (None, 274) 

def get_shape_label():
    """Get shape of the labels in Thyroid dataset"""
    return (None,)

def get_anomalous_proportion():
    return 0.15

def _get_dataset(scale):
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    data = scipy.io.loadmat("data/arrhythmia.mat")
    
    full_x_data = data["X"]
    full_y_data = data['y']
    
    x_train, x_test, \
    y_train, y_test = train_test_split(full_x_data,
                                       full_y_data,
                                       test_size=0.5,
                                       random_state=42)

    y_train = y_train.flatten().astype(int)
    y_test = y_test.flatten().astype(int)

    if scale:
        print("Scaling dataset")
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset

def _get_adapted_dataset(split, scale):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    # print("_get_adapted",scale)
    dataset = _get_dataset(scale)
    key_img = 'x_' + split
    key_lbl = 'y_' + split
    
    print("Size of split", split, ":", dataset[key_lbl].shape[0])

    return (dataset[key_img], dataset[key_lbl])

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

