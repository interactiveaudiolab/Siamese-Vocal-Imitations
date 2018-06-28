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


def configure_logger(logger):
    logger.setLevel(logging.DEBUG)
    # handlers and formatter
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('siamese.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s \t %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def configure_parser(parser):
    parser.add_argument('-c', '--cuda', action='store_const', const=True, default=False,
                        help='Whether to enable calculation on the GPU through CUDA or not. Defaults to false.')
    parser.add_argument('-s', '--spectrograms', action='store_const', const=True, default=False,
                        help='Whether to re-calculate spectrograms from the audio files or not (in which case the pre-generated .npy files are used). Defaults to false.')
    parser.add_argument('-b', '--cache_baseline', action='store_const', const=True, default=False,
                        help='Whether to use a cached version of the baseline model or not. Defaults to false.')
    parser.add_argument('-p', '--partitions', nargs=3, type=float, default=[.35, .15, .5],
                        help='Ratios by which to partition the data into training, validation, and testing sets (in that order). Defaults to [.35, .15, .5].')