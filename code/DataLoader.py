import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    if(not data_dir.endswith("/")):
        data_dir += "/"
    x_train, y_train, x_test, y_test = [], [], [], []

    for file in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]:
        data = unpickle(data_dir + file)
        x_train.append(data[b'data'])
        y_train.append(data[b'labels'])

    data = unpickle(data_dir + 'test_batch')
    x_test.append(data[b'data'])
    y_test.append(data[b'labels'])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    if (not data_dir.endswith("/")):
        data_dir += "/"
    x_test = np.load(data_dir + 'private_test_images_2022.npy')
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.95):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index = int(x_train.shape[0] * train_ratio)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

def unpickle(file):
    with open(file, 'rb') as fo:
        loaded_dict = pickle.load(fo, encoding='bytes')
    return loaded_dict

