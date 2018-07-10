import os

import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utilities


def mrr_per_epoch(train_mrrs, val_mrrs, title="MRR vs. Epoch", xlabel='epoch'):
    graph(train_mrrs, val_mrrs, title, 1, 'MRR', xlabel)


def loss_per_epoch(train_loss, val_loss, title="Loss vs. Epoch"):
    y_max = max(2, max(train_loss) + .5, max(val_loss) + .5)
    graph(train_loss, val_loss, title, y_max, 'loss', 'epoch')


def accuracy_per_epoch(train, val, title):
    graph(train, val, title, 1, 'accuracy', 'epoch')


def graph(train, val, title, y_max, y_label, x_label):
    plt.plot(train, color='blue', label='training')
    plt.plot(val, color='orange', label='validation')
    plt.legend()

    plt.ylabel(y_label)
    y_tick_interval = .1
    plt.yticks(np.arange(0, y_max + y_tick_interval, y_tick_interval))
    plt.ylim(0, y_max)

    plt.xlabel(x_label)

    plt.suptitle(title)
    plt.title("Trial #{0}".format(utilities.get_trial_number()))

    filename = title_to_filename(title)
    plt.savefig(filename)
    plt.close()


def title_to_filename(title):
    file = title
    file = file.replace(' ', '_')
    file = file.replace('.', '')
    file += '.png'
    file = file.lower()
    return os.path.join('./output', str(utilities.get_trial_number()), file)
