import os
import time

import matplotlib.pyplot as plt
import numpy as np

module_load_time = time.time()


def mrr_per_epoch(train_mrrs, val_mrrs, train_var=None, val_var=None, title="MRR vs. Epoch", xlabel='epoch'):
    plt.errorbar(np.arange(len(train_mrrs)), train_mrrs, yerr=train_var, color='blue', label='train', capsize=5, errorevery=3)
    plt.errorbar(np.arange(len(val_mrrs)), val_mrrs, yerr=val_var, color='orange', label='validation', capsize=5, errorevery=3)
    plt.legend()
    plt.ylabel('MRR')
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.title(title)
    filename = title_to_filename(title)
    plt.savefig(filename)
    plt.close()
    if train_var is not None or val_var is not None:
        mrr_per_epoch(train_mrrs, val_mrrs, title=title + " (No Error Bars)", xlabel=xlabel)


def loss_per_epoch(losses, var=None, title="Loss vs. Epoch"):
    plt.plot(losses, color='blue', label='train')
    plt.errorbar(np.arange(len(losses)), losses, yerr=var, capsize=5, errorevery=5)
    plt.legend()
    plt.ylabel('loss')
    plt.ylim(0, 1)
    plt.xlabel('epoch')
    plt.title(title)
    filename = title_to_filename(title)
    plt.savefig(filename)
    plt.close()
    if var is not None:
        loss_per_epoch(losses, title=title + " (No Error Bars)")


def title_to_filename(title):
    title = title.replace(' ', '_')
    title = title.replace('.', '')
    title += '_{0}.png'.format(module_load_time)
    title = title.lower()
    return os.path.join('./graphs', title)
