import os

import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utilities


def mean_rank_per_epoch(train, val, title_suffix, n_categories, title="Mean Rank vs. Epoch", xlabel='epoch'):
    plt.plot(train, color='blue', label='training')
    plt.plot(val, color='orange', label='validation', linestyle='dotted')
    plt.axhline(y=n_categories / 2, linestyle='dashed', color='magenta', label='random chance')

    plt.ylabel("Mean Rank")
    plt.xlabel(xlabel)

    plt.ylim(0, n_categories)

    if title_suffix:
        plt.suptitle("{0} ({1})".format(title, title_suffix))
    else:
        plt.suptitle(title)

    plt.title("Trial #{0}".format(utilities.get_trial_number()))

    plt.legend()

    filename = title_to_filename(title, title_suffix)
    plt.savefig(filename)
    plt.close()


def mrr_per_epoch(train, val, title_suffix, title="MRR vs. Epoch", xlabel='epoch', n_categories=None):
    plt.plot(train, color='blue', label='training')
    plt.plot(val, color='orange', label='validation', linestyle='dotted')
    if n_categories:
        plt.axhline(y=mrr_random_chance(n_categories), linestyle='dashed', color='magenta', label='random chance')

    plt.ylabel("MRR")
    plt.xlabel(xlabel)
    plt.ylim(0, 1)

    if title_suffix:
        plt.suptitle("{0} ({1})".format(title, title_suffix))
    else:
        plt.suptitle(title)

    plt.title("Trial #{0}".format(utilities.get_trial_number()))

    plt.legend()

    filename = title_to_filename(title, title_suffix)
    plt.savefig(filename)
    plt.close()


def loss_per_epoch(train, val, title_suffix, title="Loss vs. Epoch", log=True):
    plt.plot(train, color='blue', label='training')
    plt.plot(val, color='orange', label='validation', linestyle='dotted')

    plt.ylabel("loss")
    plt.xlabel("epoch")

    if log:
        plt.yscale('log', basey=10)
        new_title = title + " (Log)"
    else:
        y_tick_interval = .1
        y_max = max(2, max(train) + .5, max(val) + .5)
        plt.yticks(np.arange(0, y_max + y_tick_interval, y_tick_interval))
        plt.ylim(0, y_max)
        new_title = title

    if title_suffix:
        plt.suptitle("{0} ({1})".format(new_title, title_suffix))
    else:
        plt.suptitle(title)

    plt.title("Trial #{0}".format(utilities.get_trial_number()))

    plt.legend()

    filename = title_to_filename(new_title, title_suffix)
    plt.savefig(filename)
    plt.close()

    if log:
        loss_per_epoch(train, val, title_suffix, title=title, log=False)


def accuracy_per_epoch(train, val, title_suffix, title="Accuracy vs. Epoch"):
    plt.plot(train, color='blue', label='training')
    plt.plot(val, color='orange', label='validation', linestyle='dotted')

    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.ylim(0, 1)

    if title_suffix:
        plt.suptitle("{0} ({1})".format(title, title_suffix))
    else:
        plt.suptitle(title)

    plt.title("Trial #{0}".format(utilities.get_trial_number()))

    plt.legend()

    filename = title_to_filename(title, title_suffix)
    plt.savefig(filename)
    plt.close()


def title_to_filename(title, suffix):
    if suffix:
        file = suffix + '_' + title
    else:
        file = title
    file = file.replace(' ', '_').replace('.', '').replace(',', '')
    file += '.png'
    file = file.lower()
    return os.path.join('./output', str(utilities.get_trial_number()), file)


def mrr_random_chance(n_categories):
    return np.mean([1 / n for n in np.random.randint(low=1, high=n_categories + 1, size=99999)])
