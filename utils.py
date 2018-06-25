import os

import numpy as np
import torch
import logging


def load_model(model, base_path):
    model.load_state_dict(torch.load(base_path))


def save_model(model, base_path):
    torch.save(model.state_dict(), base_path)


def load_npy(name):
    return np.load(os.environ['SIAMESE_DATA_DIR'] + "/npy/" + name)


def save_npy(array, suffix, ar_type=None):
    array = np.array(array)
    if ar_type:
        array = array.astype(ar_type)
    np.save(os.environ['SIAMESE_DATA_DIR'] + "/npy/" + suffix, array)


def prindent(str, n_indent):
    logger = logging.getLogger('logger')
    p_str = ''.join(['\t' for i in range(n_indent)]) + str
    logger.info(p_str)


def log_final_stats(rrs):
    prindent("mean: {0}".format(rrs.mean()), 1)
    prindent("stddev: {0}".format(rrs.std()), 1)
    prindent("min: {0}".format(rrs.min()), 1)
    prindent("max: {0}".format(rrs.max()), 1)


def np_index_of(array, item):
    """
    Find the first index of item in array
    :param array: ndarray
    :param item:
    :return: first index of item in array
    """
    where = np.where(array == item)
    if len(where) == 0:
        raise ValueError("{0} not found in array".format(item))
    return where[0][0]  # where is a 2d array
