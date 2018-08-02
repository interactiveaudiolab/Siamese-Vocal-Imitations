import os

import numpy as np

import utils.utils as utilities


def loss_rank_overlay(loss, rank, left_ax, title, loss_max, rank_max, correlation):
    correlation_message = "correlation = {0}".format(np.round(correlation, 2))

    l1 = left_ax.plot(loss, color='blue', label='loss')
    left_ax.set_title(title + " ({0})".format(correlation_message))
    left_ax.set_ylabel("loss", color='blue')
    left_ax.tick_params('y', colors='blue')
    left_ax.set_ylim(0, loss_max)
    # left_ax.text(len(loss) / 2, .5 * loss_max, correlation_message, bbox=dict(facecolor='black', alpha=0.2))

    right_ax = left_ax.twinx()
    l2 = right_ax.plot(rank, color='orange', label='rank')
    right_ax.set_ylabel("rank", color="orange")
    right_ax.tick_params('y', colors='orange')
    right_ax.set_ylim(0, rank_max)

    left_ax.legend(l1 + l2, [l.get_label() for l in (l1 + l2)], loc=0)


def mean_rank_per_epoch(train, val, n_categories, ax, title="Mean Rank vs. Epoch", xlabel='epoch'):
    ax.plot(train, color='blue', label='training')
    ax.plot(val, color='orange', label='validation', linestyle='dotted')
    if n_categories:
        ax.axhline(y=n_categories / 2, linestyle='dashed', color='magenta', label='random chance')

    ax.set_ylabel("Mean Rank")
    ax.set_xlabel(xlabel)

    ax.set_ylim(0, n_categories)

    ax.set_title(title)

    ax.legend()


def mrr_per_epoch(train, val, ax, title="MRR vs. Epoch", xlabel='epoch', n_categories=None):
    ax.plot(train, color='blue', label='training')
    ax.plot(val, color='orange', label='validation', linestyle='dotted')
    if n_categories:
        ax.axhline(y=mrr_random_chance(n_categories), linestyle='dashed', color='magenta', label='random chance')

    ax.set_ylabel("MRR")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1)

    ax.set_title(title)

    ax.legend()


def loss_per_epoch(train, val, ax, title="Loss vs. Epoch", log=True):
    ax.plot(train, color='blue', label='training')
    ax.plot(val, color='orange', label='validation', linestyle='dotted')

    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")

    if log:
        ax.set_yscale('log', basey=10)
        title += " (Log)"
    else:
        y_tick_interval = .1
        y_max = max(2, max(train) + .5, max(val) + .5)
        ax.set_yticks(np.arange(0, y_max + y_tick_interval, y_tick_interval))
        ax.set_ylim(0, y_max)

    ax.set_title(title)

    ax.legend()


def accuracy_per_epoch(train, val, ax, title="Accuracy vs. Epoch"):
    ax.plot(train, color='blue', label='training')
    ax.plot(val, color='orange', label='validation', linestyle='dotted')

    ax.set_ylabel("accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)

    ax.set_title(title)
    ax.legend()


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
