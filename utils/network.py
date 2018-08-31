import logging
import pathlib

import torch

from models.siamese import Siamese
from utils.utils import get_trial_directory


def load_model(model, path, use_cuda=True):
    if use_cuda:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


def save_model(model, path):
    try:
        torch.save(model.state_dict(), path)
    except FileNotFoundError:
        pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
        save_model(model, path)


def initialize_siamese_params(regenerate, dropout):
    logger = logging.getLogger('logger')
    starting_weights_path = "./output/models/siamese_init/starting_weights"

    model = Siamese(dropout=dropout)
    if not regenerate:
        load_model(model, starting_weights_path)

    logger.debug("Saving initial weights/biases at {0}...".format(starting_weights_path))
    save_model(model, starting_weights_path)

    trial_path = get_trial_directory('init_weights')
    logger.debug("Saving initial weights/biases at {0}...".format(trial_path))
    save_model(model, trial_path)


def initialize_weights(siamese, use_cuda):
    logger = logging.getLogger('logger')

    starting_weights_path = "./output/models/siamese_init/{0}".format("starting_weights")

    try:
        logger.debug("Loading initial weights/biases from {0}...".format(starting_weights_path))
        load_model(siamese, starting_weights_path, use_cuda)
    except FileNotFoundError:
        logger.debug("Saving initial weights/biases at {0}...".format(starting_weights_path))
        save_model(siamese, starting_weights_path)

    return siamese
