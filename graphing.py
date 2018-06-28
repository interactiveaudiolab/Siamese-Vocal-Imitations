import _tkinter
import logging

import matplotlib.pyplot as plt


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
            title.replace(' ', '')
            title.replace('.', '')
            title += '.png'
        plt.savefig(title)
    except _tkinter.TclError:
        logger.warning("No xwindows or xwindows forwarding, not graphing. Did you remember to log into the ssh server using the -Y flag?")
