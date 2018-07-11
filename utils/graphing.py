import os

import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utilities


def mrr_per_epoch(train_mrrs, val_mrrs, title_suffix, title="MRR vs. Epoch", xlabel='epoch'):
    graph(train_mrrs, val_mrrs, title, 1, 'MRR', xlabel, title_suffix)


def loss_per_epoch(train_loss, val_loss, title_suffix, title="Loss vs. Epoch"):
    y_max = max(2, max(train_loss) + .5, max(val_loss) + .5)
    graph(train_loss, val_loss, title, y_max, 'loss', 'epoch', title_suffix)


def accuracy_per_epoch(train, val, title_suffix, title="Accuracy vs. Epoch"):
    graph(train, val, title, 1, 'accuracy', 'epoch', title_suffix)


def graph(train, val, title, y_max, y_label, x_label, title_suffix):
    plt.plot(train, color='blue', label='training')
    plt.plot(val, color='orange', label='validation')
    plt.legend()

    plt.ylabel(y_label)
    y_tick_interval = .1
    plt.yticks(np.arange(0, y_max + y_tick_interval, y_tick_interval))
    plt.ylim(0, y_max)

    plt.xlabel(x_label)

    plt.suptitle(title + " ({0})".format(title_suffix))
    plt.title("Trial #{0}".format(utilities.get_trial_number()))

    filename = title_to_filename(title, title_suffix)
    plt.savefig(filename)
    plt.close()


def title_to_filename(title, suffix):
    file = suffix + '_' + title
    file = file.replace(' ', '_').replace('.', '').replace(',', '')
    file += '.png'
    file = file.lower()
    return os.path.join('./output', str(utilities.get_trial_number()), file)
