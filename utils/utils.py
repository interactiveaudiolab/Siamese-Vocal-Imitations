import logging
import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import yaml


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


def configure_logger(logger, console=True, file=True):
    # Remove any old handlers that may exist
    old_handlers = []
    for handler in logger.handlers:
        old_handlers.append(handler)
    for handler in old_handlers:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s \t %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    if file:
        file_handler = logging.FileHandler(os.path.join())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def configure_parser(parser):
    general = parser.add_argument_group(title="General options")
    general.add_argument('-c', '--cuda', action='store_const', const=True, default=False,
                         help='Whether to enable calculation on the GPU through CUDA or not. Defaults to false.')
    general.add_argument('-tr', '--trials', default=1, type=int,
                         help='Amount of trials to run. Defaults to 1.')
    general.add_argument('-vf', '--validation_frequency', default=1, type=int,
                         help='Frequency of MRR and validation loss calculations (per epoch). Defaults to 1. 0 means do not calculate at all.')
    general.add_argument('-e', '--epochs', type=int, default=300, help="Amount of epochs to train for. Defaults to 300")

    network = parser.add_argument_group(title="Network options")
    network.add_argument('-p', '--pairwise', action='store_const', const=True, default=False,
                         help='Train a model with a pairwise loss function.')
    network.add_argument('-t', '--triplet', action='store_const', const=True, default=False,
                         help='Train a model with a triplet loss function.')
    network.add_argument('-dr', '--dropout', action='store_const', const=True, default=False,
                         help='Whether to use drop-out in training the network. Defaults to false.')
    network.add_argument('-rw', '--regenerate_weights', action='store_const', const=True, default=False,
                         help='Whether to regenerate the initial network weights or not. Defaults to false.')

    optimizer = parser.add_argument_group(title="Optimizer options")
    optimizer.add_argument('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'],
                           help='Optimizer to use. Defaults to adam.')
    optimizer.add_argument('-lr', '--learning_rate', type=float, default=.001,
                           help='Learning rate. Defaults to .001')
    optimizer.add_argument('-wd', '--weight_decay', type=float, default=0,
                           help='Weight decay. Defaults to 0.')
    optimizer.add_argument('-m', '--momentum', action='store_const', const=True, default=False,
                           help='Whether to use momentum. Only applies when using SGD or RMSProp. Defaults to false.')

    data = parser.add_argument_group(title="Data options")
    data.add_argument('-pr', '--partitions', nargs=3, type=float, default=[.35, .15, .5],
                      help='Ratios by which to partition the data into training, validation, and testing sets (in that order). Defaults to [.35, .15, .5].')
    data.add_argument('-d', '--dataset', default='vi', type=str, choices=['vs1.0', 'vs1.1', 'vi'],
                      help='Dataset to use for experiments. Defaults to vocal imitation.')
    data.add_argument('-rs', '--regenerate_splits', action='store_const', const=True, default=False,
                      help='Whether to regenerate data splits or not. Defaults to false.')
    data.add_argument('--recalculate_spectrograms', action='store_const', const=True, default=False,
                      help='Whether to re-calculate spectrograms from the audio files or not (in which case the pre-generated .npy files are used). Defaults to false.')
    data.add_argument('-nc', '--num_categories', default=None, type=int,
                      help='Fixed number of categories to use in the training/validation set. Defaults to none, in which case the amount of categories will '
                           'be determined based on the partitions.')


def get_trial_directory(suffix=''):
    return os.path.join("./output/trials/{0}".format(get_trial_number()), suffix)


def update_trial_number():
    trial_number = get_trial_number()
    with open('./output/state/state.pickle', 'wb') as handle:
        pickle.dump(trial_number + 1, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_trial_number():
    try:
        with open('./output/state/state.pickle', 'rb') as handle:
            trial_number = pickle.load(handle)
    except FileNotFoundError:
        with open('./output/state/state.pickle', 'w+b') as handle:
            trial_number = 0
            pickle.dump(trial_number + 1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return trial_number


def create_output_directory():
    trial_number = get_trial_number()
    pathlib.Path('./output/trials/{0}'.format(trial_number)).mkdir(exist_ok=True, parents=True)


def zip_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_dataset_dir(dataset):
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
            try:
                return config['datasets'][dataset]
            except KeyError:
                logger = logging.getLogger('logger')
                logger.critical("No location for datasets.{0} specified in config.yaml.".format(dataset))
                sys.exit()
    except FileNotFoundError:
        logger = logging.getLogger('logger')
        logger.critical("No config.yaml file found. Please generate one in the top level directory.")
        sys.exit()


def get_npy_dir(dataset):
    try:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
            try:
                return os.path.join(config['spectrogram_cache_location'], dataset)
            except KeyError:
                logger = logging.getLogger('logger')
                logger.critical("No location to save spectrograms specified on disk.")
                sys.exit()
    except FileNotFoundError:
        logger = logging.getLogger('logger')
        logger.critical("No config.yaml file found. Please generate one in the top level directory.")
        sys.exit()


def get_optimizer(network, optimizer_name, lr, wd, momentum):
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, weight_decay=wd, momentum=.9 if momentum else 0, nesterov=momentum)  # TODO: separate params
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(), lr=lr, weight_decay=wd, momentum=.9 if momentum else 0)
    else:
        raise ValueError("No optimizer found with name {0}".format(optimizer_name))
    return optimizer
