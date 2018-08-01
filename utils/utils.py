import logging
import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import yaml

from models.siamese import Siamese


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
    general = parser.add_argument_group(title="General options")
    general.add_argument('-c', '--cuda', action='store_const', const=True, default=False,
                         help='Whether to enable calculation on the GPU through CUDA or not. Defaults to false.')
    general.add_argument('-tr', '--trials', default=1, type=int,
                         help='Amount of trials to run. Defaults to 1.')
    general.add_argument('-vf', '--validation_frequency', default=1, type=int,
                         help='Frequency of MRR and validation loss calculations (per epoch). Defaults to 1. 0 means do not calculate at all.')
    general.add_argument('-e', '--epochs', type=int, default=100, help="Amount of epochs to train for. Defaults to 100")
    general.add_argument('--no_test', action='store_const', const=True, default=False,
                         help="Whether to skip calculating MRR on the testing set at the end of training. Defaults to false.")

    network = parser.add_argument_group(title="Network options")
    network.add_argument('-s', '--siamese', action='store_const', const=True, default=False,
                         help='Use a siamese network.')
    network.add_argument('-t', '--triplet', action='store_const', const=True, default=False,
                         help='Use a triplet network.')
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
    data.add_argument('-p', '--partitions', nargs=3, type=float, default=[.35, .15, .5],
                      help='Ratios by which to partition the data into training, validation, and testing sets (in that order). Defaults to [.35, .15, .5].')
    data.add_argument('-d', '--dataset', default='vi', type=str, choices=['vs1.0', 'vs2.0', 'vi'],
                      help='Dataset to use for experiments. Defaults to vocal imitation.')
    data.add_argument('-rs', '--regenerate_splits', action='store_const', const=True, default=False,
                      help='Whether to regenerate data splits or not. Defaults to false.')
    data.add_argument('--recalculate_spectrograms', action='store_const', const=True, default=False,
                      help='Whether to re-calculate spectrograms from the audio files or not (in which case the pre-generated .npy files are used). Defaults to false.')
    data.add_argument('-nc', '--num_categories', default=None, type=int,
                      help='Fixed number of categories to use in the training/validation set. Defaults to none, in which case the amount of categories will '
                           'be determined based on the partitions.')


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


def initialize_siamese_params(regenerate, dropout):
    logger = logging.getLogger('logger')
    starting_weights_path = "./model_output/siamese_init/starting_weights"

    model = Siamese(dropout=dropout)
    if not regenerate:
        load_model(model, starting_weights_path)

    logger.debug("Saving initial weights/biases at {0}...".format(starting_weights_path))
    save_model(model, starting_weights_path)

    trial_path = "./output/{0}/init_weights".format(get_trial_number())
    logger.debug("Saving initial weights/biases at {0}...".format(trial_path))
    save_model(model, trial_path)


def initialize_weights(siamese, use_cuda):
    logger = logging.getLogger('logger')

    starting_weights_path = "./model_output/siamese_init/{0}".format("starting_weights")

    try:
        logger.debug("Loading initial weights/biases from {0}...".format(starting_weights_path))
        load_model(siamese, starting_weights_path, use_cuda)
    except FileNotFoundError:
        logger.debug("Saving initial weights/biases at {0}...".format(starting_weights_path))
        save_model(siamese, starting_weights_path)

    return siamese


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
