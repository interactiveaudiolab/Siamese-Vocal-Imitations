import logging
import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import yaml


def load_model(model, path, use_cuda=True):
    if use_cuda:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_npy(name, prefix):
    path = os.path.join(get_dataset_dir(), "npy", prefix, name)
    return np.load(path)


def save_npy(array, name, prefix, ar_type=None):
    array = np.array(array)
    if ar_type:
        array = array.astype(ar_type)
    path = os.path.join(get_dataset_dir(), "npy", prefix, name)
    try:
        np.save(path, array)
    except FileNotFoundError:  # can occur when the parent directory doesn't exist
        os.mkdir(os.path.dirname(path))
        np.save(path, array)


def prindent(string, n_indent):
    logger = logging.getLogger('logger')
    p_str = ''.join(['\t' for _ in range(n_indent)]) + string
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


def configure_logger(logger, console_only=False):
    # Remove any old handlers that may exist
    old_handlers = []
    for handler in logger.handlers:
        old_handlers.append(handler)
    for handler in old_handlers:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s \t %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    if not console_only:
        file_handler = logging.FileHandler('./output/{0}/siamese.log'.format(get_trial_number()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def configure_parser(parser):
    parser.add_argument('-c', '--cuda', action='store_const', const=True, default=False,
                        help='Whether to enable calculation on the GPU through CUDA or not. Defaults to false.')
    parser.add_argument('-s', '--spectrograms', action='store_const', const=True, default=False,
                        help='Whether to re-calculate spectrograms from the audio files or not (in which case the pre-generated .npy files are used). Defaults to false.')
    parser.add_argument('-b', '--cache_baseline', action='store_const', const=True, default=False,
                        help='Whether to use a cached version of the baseline model or not. Defaults to false.')
    parser.add_argument('-p', '--data_partitions', nargs=3, type=float, default=[.35, .15, .5],
                        help='Ratios by which to partition the data into training, validation, and testing sets (in that order). Defaults to [.35, .15, .5].')
    parser.add_argument('-f', '--fine_tuning_passes', type=int, default=0,
                        help='Minimum amount of fine tuning passes to perform, regardless of convergence. Defaults to 0.')
    parser.add_argument('-t', '--trials', default=1, type=int,
                        help='Amount of trials to run. Defaults to 1.')
    parser.add_argument('-r', '--random_only', action='store_const', const=True, default=False,
                        help='Whether to only run the random selection phase of training and skip fine-tuning. Defaults fo false.')
    parser.add_argument('-d', '--dropout', action='store_const', const=True, default=False,
                        help='Whether to use drop-out in training the Siamese network. Defaults to false.')
    parser.add_argument('-sd', '--siamese_dataset', default='vs2.0', type=str, choices=['vs1.0', 'vs2.0', 'vi'],
                        help='Siamese dataset to use for experiments. Defaults to vocal sketch 2.0.')
    parser.add_argument('-ve', '--validate_every', default=1, type=int,
                        help='How often to calculate loss and MRR (per epoch). Defaults to 1. 0 means do not calculate at all.')
    parser.add_argument('-rs', '--regenerate_splits', action='store_const', const=True, default=False,
                        help='Whether to regenerate data splits or not. Defaults to false.')
    parser.add_argument('-rw', '--regenerate_weights', action='store_const', const=True, default=False,
                        help='Whether to regenerate Siamese weights or not. Defaults to false.')


def update_trial_number():
    trial_number = get_trial_number()
    with open('state.pickle', 'wb') as handle:
        pickle.dump(trial_number + 1, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_trial_number():
    try:
        with open('state.pickle', 'rb') as handle:
            trial_number = pickle.load(handle)
    except FileNotFoundError:
        with open('state.pickle', 'w+b') as handle:
            trial_number = 0
            pickle.dump(trial_number + 1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return trial_number


def create_output_directory():
    trial_number = get_trial_number()
    pathlib.Path('./output/{0}'.format(trial_number)).mkdir(exist_ok=True)


def zip_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_dataset_dir():
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
            try:
                return config['SIAMESE_DATA_DIR']
            except KeyError:
                logger = logging.getLogger('logger')
                logger.critical("No entry for SIAMESE_DATA_DIR found in config.yaml.")
                sys.exit()
    except FileNotFoundError:
        logger = logging.getLogger('logger')
        logger.critical("No config.yaml file found. Please generate one in the top level directory.")
        sys.exit()
