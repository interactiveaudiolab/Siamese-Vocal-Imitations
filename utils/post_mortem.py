import csv
import os
import pickle
import shutil
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

import utils.graphing
from utils.obj import TrainingProgress


def load_training_result(path):
    progress = TrainingProgress()
    progress.load(path)
    return progress


def get_correlations(directory):
    correlations = []
    join = os.path.join(directory, "*/")
    for folder in glob(join):
        trial_no = int(os.path.basename(os.path.dirname(folder)))

        siamese_path = os.path.join(folder, 'siamese.pickle')
        siamese_result = load_training_result(siamese_path)

        triplet_path = os.path.join(folder, 'triplet.pickle')
        triplet_result = load_training_result(triplet_path)
        if siamese_result and triplet_result:
            siamese_tr, siamese_vl = siamese_result.pearson()
            triplet_tr, triplet_vl = triplet_result.pearson()
            correlations.append([trial_no, siamese_tr, siamese_vl, triplet_tr, triplet_vl])

    # sort by trial number
    correlations.sort(key=lambda c: c[0])
    return np.array(correlations)


def correlation_csv(directory, correlations):
    csv_path = os.path.join(directory, "correlations.csv")
    with open(csv_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_no', 'siamese_tr', 'siamese_vl', 'triplet_tr', 'triplet_vl'])
        for b in correlations:
            writer.writerow(b)
        averages = np.mean(correlations[:, 1:], axis=0)
        writer.writerow(np.concatenate((["AVERAGE"], averages)))
        std_devs = np.std(correlations[:, 1:], axis=0)
        writer.writerow(np.concatenate((["STD_DEV"], std_devs)))


def boxplot(directory, correlations, p_value):
    # first and third columns are the training correlations for siamese and triplet nets respectively
    correlations = correlations[:, [1, 3]]

    # Multiple box plots on one Axes
    fig, ax = plt.subplots()
    fig.set_size_inches(9.5, 6)
    d = ax.boxplot(correlations)
    n = len(correlations)
    ax.set_xticklabels(["Pairwise", "Triplet"], fontsize=18)
    ax.tick_params('y', labelsize=18)
    medians = [median.get_xydata() for median in d['medians']]

    ax.text(.5, .25, "p = {0}".format(np.round(p_value, 3)), horizontalalignment='center', transform=ax.transAxes, fontsize=18)
    ax.text(.5, .15, "n = {0}".format(n), horizontalalignment='center', transform=ax.transAxes, fontsize=18)
    for median in medians:
        x_right = median[1][0]
        y = median[0][1]
        ax.text(x_right + .01, y, "{0}".format(np.round(y, 2)), horizontalalignment='left', verticalalignment='center', fontsize=18)
    ax.set_ylim(-1, 1)
    ax.set_title("Correlation between loss over time and rank over time", fontsize=18)
    ax.set_ylabel("Pearson's correlation coefficient", fontsize=18)
    ax.set_xlabel("Loss", fontsize=18)
    plt.savefig(os.path.join(directory, "correlation_boxplot.pdf"))
    plt.close()


def condense_graphs(directory, verbose=False):
    for file in glob(os.path.join(directory, "*/*")):
        if file.endswith("png"):
            path = Path(file)
            dest_name = path.parents[0].name + "_" + path.name
            p = os.path.join(path.parents[1], dest_name)
            if verbose:
                print("{0} --> {1}".format(path, p))
            shutil.copyfile(path, p)


def loss_rank_overlay(directory, trial_no):
    path = os.path.join(directory, str(trial_no))
    siamese = load_training_result(os.path.join(path, "siamese.pickle"))
    triplet = load_training_result(os.path.join(path, "triplet.pickle"))

    fig, (ax1, ax2) = plt.subplots(2, 1)

    fig.set_size_inches(10, 8)

    utils.graphing.loss_rank_overlay(siamese.train_loss, siamese.train_rank, ax1, "Pairwise", siamese.pearson()[0])
    utils.graphing.loss_rank_overlay(triplet.train_loss, triplet.train_rank, ax2, "Triplet", triplet.pearson()[0])

    # fig.suptitle("Loss vs. Rank, Trial #{0}".format(trial_no), y=1, fontsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "loss_rank_overlay.pdf"), dpi=180)
    plt.close()


def wilcox_test(correlations):
    diff, p = wilcoxon(correlations[:, 1], correlations[:, 3])
    average_diff = diff / len(correlations)
    return average_diff, p


def get_representative_trial(correlations):
    """
    Find the most representative trial by calculating sum of squared differences between correlation and mean correlation.

    :param correlations:
    :return:
    """
    means = np.mean(correlations, axis=0)
    deviations = (correlations - means)[:, (1, 3)]
    ssd = np.sum(deviations ** 2, axis=1)
    m = np.argmin(ssd)

    return int(correlations[m, 0])
