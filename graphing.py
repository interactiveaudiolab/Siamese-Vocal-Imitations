import _tkinter
import functools
import logging

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
        except _tkinter.TclError:
            logger.warning("No xwindows or xwindows forwarding, not graphing. Did you remember to log into the ssh server using the -Y flag?")

    return wrapper


@xwindows_exception
def mrr_per_epoch(train_mrrs, val_mrrs, title="MRR vs. Epoch"):
    plt.plot(train_mrrs, color='blue', label='train')
    plt.plot(val_mrrs, color='orange', label='validation')
    plt.legend()
    plt.ylabel('MRR')
    plt.xlabel('epoch')
    plt.title(title)
    title = title_to_filename(title)
    plt.savefig(title)


@xwindows_exception
def loss_per_epoch(losses, var, title="Loss vs. Epoch"):
    plt.plot(losses, color='blue', label='train')
    plt.errorbar(np.arange(len(losses)), losses, yerr=var, capsize=5)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(title)
    title = title_to_filename(title)
    plt.savefig(title)


def title_to_filename(title):
    title = title.replace(' ', '_')
    title = title.replace('.', '')
    if not title.endswith('.png'):
        title += '.png'
    return title
