import os

import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utilities


def mrr_per_epoch(train_mrrs, val_mrrs, title_suffix, title="MRR vs. Epoch", xlabel='epoch', n_categories=None):
    if n_categories:
        graph(train_mrrs, val_mrrs, title, 1, 'MRR', xlabel, title_suffix, hline=random_chance(n_categories))
    else:
        graph(train_mrrs, val_mrrs, title, 1, 'MRR', xlabel, title_suffix)


def loss_per_epoch(train_loss, val_loss, title_suffix, title="Loss vs. Epoch"):
    y_max = max(2, max(train_loss) + .5, max(val_loss) + .5)
    graph(train_loss, val_loss, title, 2, 'loss', 'epoch', title_suffix, log=True)


def accuracy_per_epoch(train, val, title_suffix, title="Accuracy vs. Epoch"):
    graph(train, val, title, 1, 'accuracy', 'epoch', title_suffix)


def graph(train, val, title, y_max, y_label, x_label, title_suffix, log=False, hline=None):
    plt.plot(train, color='blue', label='training')
    plt.plot(val, color='orange', label='validation')

    plt.ylabel(y_label)

    if log:
        plt.yscale('log', basey=10)
        new_title = title + " (Log)"
    else:
        y_tick_interval = .1
        plt.yticks(np.arange(0, y_max + y_tick_interval, y_tick_interval))
        plt.ylim(0, y_max)
        new_title = title

    if hline:
        plt.axhline(y=hline, linestyle='dashed', color='magenta', label='random chance')

    plt.xlabel(x_label)

    plt.suptitle("{0} ({1})".format(title, title_suffix))
    plt.title("Trial #{0}".format(utilities.get_trial_number()))

    plt.legend()

    filename = title_to_filename(new_title, title_suffix)
    plt.savefig(filename)
    plt.close()

    if log:
        graph(train, val, title, y_max, y_label, x_label, title_suffix, log=False)


def title_to_filename(title, suffix):
    file = suffix + '_' + title
    file = file.replace(' ', '_').replace('.', '').replace(',', '')
    file += '.png'
    file = file.lower()
    return os.path.join('./output', str(utilities.get_trial_number()), file)


def chance_by_category():
    x = np.arange(120) + 1
    y = [random_chance(p) for p in x]
    plt.plot(x, y)
    plt.title("Random Chance MRR vs. Number of Categories")
    plt.ylabel("MRR")
    plt.xlabel("# categories")
    plt.savefig("random_mrr.png")
    plt.close()


def random_chance(n_categories):
    return np.mean([1 / n for n in np.random.randint(low=1, high=n_categories + 1, size=99999)])
