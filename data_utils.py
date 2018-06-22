import os

import numpy as np
import torch
import logging


def load_model_from_epoch(model, best_epoch, base_path, path_suffix):
    format_str = "{0}_{1}".format(path_suffix, best_epoch)
    load_model(base_path, model, format_str)


def load_model(base_path, model, path_suffix):
    model.load_state_dict(torch.load(base_path.format(path_suffix)))


def save_model(base_path, model, path_suffix):
    torch.save(model.state_dict(), base_path.format(path_suffix))


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
