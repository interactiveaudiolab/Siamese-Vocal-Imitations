import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import Formatter, LogFormatter

import utils.utils as utilities


class ConciseScientificNotationFormatter(Formatter):
    def __call__(self, x, pos=None):
        # if exponent is 0, don't bother with scientific notation
        if 1 <= x < 10:
            return str(x).rstrip('0').rstrip('.')
        else:
            # get python's string formatter version of scientific notation
            verbose = '%E' % x

            # strip off trailing zeros that might be on the mantissa, and if it's an integer, the decimal point too
            mantissa = verbose.split('E')[0]
            mantissa = mantissa.rstrip('0').rstrip('.')

            # omit the exponent's sign if it's positive and strip leading zeros off
            exponent = verbose.split('E')[1]
            exponent_sign = '-' if '-' in exponent else ''
            exponent = exponent[1:].lstrip('0')
            return mantissa + 'e' + exponent_sign + exponent


def loss_rank_overlay(loss, rank, left_ax, title, correlation, font_size=18):
    correlation_message = "correlation = {0}".format(np.round(correlation, 2))
    color1 = [c / 255 for c in (20, 46, 120)]
    color2 = [c / 255 for c in (255, 87, 20)]

    l1 = left_ax.plot(loss, color=color1, label='loss')
    left_ax.set_title(title, fontsize=font_size)
    left_ax.set_yscale('log', basey=10)
    left_ax.set_ylabel("loss", color=color1, fontsize=font_size)
    left_ax.set_ylim(top=1)
    left_ax.tick_params('y', colors=color1, which='both', labelsize=font_size)
    left_ax.tick_params('x', labelsize=font_size)

    left_ax.text(.5, .85, correlation_message, bbox=dict(facecolor='black', alpha=0.2), horizontalalignment='center',
                 fontsize=font_size, transform=left_ax.transAxes)
    left_ax.yaxis.set_major_formatter(LogFormatter())
    left_ax.yaxis.set_minor_formatter(LogFormatter())
    left_ax.minorticks_off()

    right_ax = left_ax.twinx()
    l2 = right_ax.plot(rank, color=color2, label='rank', linestyle='dashed')
    right_ax.set_yscale('log', basey=10)
    right_ax.set_ylabel("rank", color=color2, fontsize=font_size)
    right_ax.tick_params('y', colors=color2, which='both', labelsize=font_size)
    right_ax.yaxis.set_major_formatter(LogFormatter())
    right_ax.yaxis.set_minor_formatter(LogFormatter())
    # right_ax.minorticks_off()
    right_ax.set_ylim(bottom=1)

    left_ax.legend(l1 + l2, [l.get_label() for l in (l1 + l2)], loc=1, fontsize=font_size)


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


def correlation_boxplot(directory, data, p_value):
    # Multiple box plots on one Axes
    fig, ax = plt.subplots()
    fig.set_size_inches(9.5, 6)
    d = ax.boxplot(data)
    n = len(data)
    ax.set_xticklabels(["Pairwise", "Triplet"], fontsize=18)
    ax.tick_params('y', labelsize=18)
    medians = [median.get_xydata() for median in d['medians']]
    ax.text(.5, .25, "p = {0}".format(np.round(p_value, 6)), horizontalalignment='center', transform=ax.transAxes,
            fontsize=18)
    ax.text(.5, .15, "n = {0}".format(n), horizontalalignment='center', transform=ax.transAxes, fontsize=18)
    for median in medians:
        x_right = median[1][0]
        y = median[0][1]
        ax.text(x_right + .01, y, "{0}".format(np.round(y, 2)), horizontalalignment='left', verticalalignment='center',
                fontsize=18)
    ax.set_ylim(-.25, 1)
    ax.set_title("Correlation between loss over time and rank over time", fontsize=18)
    ax.set_ylabel("Pearson's correlation coefficient", fontsize=18)
    ax.set_xlabel("Loss", fontsize=18)
    plt.savefig(os.path.join(directory, "correlation_boxplot.pdf"))
    plt.close()


def num_canonical_memorized(memorized_var, n_memorized, evaluated_epochs, num_categories):
    plt.plot(evaluated_epochs, n_memorized, color='b', label='performance')
    plt.axhline(y=num_categories, color='magenta', label='# categories')
    plt.ylabel('# of categories memorized')
    plt.xlabel('epoch')
    plt.title("# of Categories Memorized vs. Time")
    plt.savefig('n_memorized.png')
    plt.close()

    plt.plot(evaluated_epochs, memorized_var, color='b', label='variance across imitations')
    plt.ylabel("variance")
    plt.xlabel('epoch')
    plt.title("Variance across Imitations vs. Time")
    plt.savefig("variance.png")
    plt.close()
