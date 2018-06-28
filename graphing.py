import _tkinter
import logging

import matplotlib.pyplot as plt
import numpy as np


def mrr_per_epoch(train_mrrs, val_mrrs, title="MRR vs. Epoch"):
    logger = logging.getLogger('logger')
    try:
        plt.plot(train_mrrs, color='blue', label='train')
        plt.plot(val_mrrs, color='orange', label='validation')
        plt.legend()
        plt.ylabel('MRR')
        plt.xlabel('epoch')
        plt.title(title)
        if not title.endswith('.png'):
            title = title.replace(' ', '_')
            title = title.replace('.', '')
            title += '.png'
        plt.savefig(title)
    except _tkinter.TclError:
        logger.warning("No xwindows or xwindows forwarding, not graphing. Did you remember to log into the ssh server using the -Y flag?")


def loss_per_epoch(losses, var, title="Loss vs. Epoch"):
    logger = logging.getLogger('logger')
    try:
        plt.plot(losses, color='blue', label='train')
        plt.errorbar(np.arange(len(losses)), losses, yerr=var, capsize=5)
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title(title)
        if not title.endswith('.png'):
            title = title.replace(' ', '_')
            title = title.replace('.', '')
            title += '.png'
        plt.savefig(title)
    except _tkinter.TclError:
        logger.warning("No xwindows or xwindows forwarding, not graphing. Did you remember to log into the ssh server using the -Y flag?")
