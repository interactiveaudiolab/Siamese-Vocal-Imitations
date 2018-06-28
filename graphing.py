import functools
import logging
import os
import time
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np


def xwindows_exception(f):
    """
    Defines a wrapper that handles exceptions where plt.plot() fails due to a lack of a valid xwindows.

    :param f: function to wrap
    :return:
    """
    logger = logging.getLogger('logger')

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TclError:
            logger.warning("No xwindows or xwindows forwarding, not graphing. Did you remember to log into the ssh server using the -Y flag?")

    return wrapper


@xwindows_exception
def mrr_per_epoch(train_mrrs, val_mrrs, train_var=None, val_var=None, title="MRR vs. Epoch"):
    plt.errorbar(np.arange(len(train_mrrs)), train_mrrs, yerr=train_var, color='blue', label='train', capsize=5, errorevery=3)
    plt.errorbar(np.arange(len(val_mrrs)), val_mrrs, yerr=val_var, color='orange', label='validation', capsize=5, errorevery=3)
    plt.legend()
    plt.ylabel('MRR')
    plt.xlabel('epoch')
    plt.title(title)
    title = title_to_filename(title)
    plt.savefig(title)
    plt.close()


@xwindows_exception
def loss_per_epoch(losses, var, title="Loss vs. Epoch"):
    plt.plot(losses, color='blue', label='train')
    plt.errorbar(np.arange(len(losses)), losses, yerr=var, capsize=5, errorevery=5)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(title)
    title = title_to_filename(title)
    plt.savefig(title)
    plt.close()


def title_to_filename(title):
    title = title.replace(' ', '_')
    title = title.replace('.', '')
    title += '_{0}.png'.format(time.time())
    title = title.lower()
    return os.path.join('./graphs', title)
