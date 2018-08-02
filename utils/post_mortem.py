import csv
import os
import pickle
import shutil
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from utils.graphing import loss_rank_overlay


def load_training_result(path):
    try:
        with open(path, 'rb') as f:
            result = pickle.load(f)
            return result
    except FileNotFoundError:
        pass


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


def generate_correlation_csv(directory, correlations):
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


def generate_boxplot(directory, correlations):
    # first and third columns are the training correlations for siamese and triplet nets respectively
    correlations = correlations[:, [1, 3]]

    # Multiple box plots on one Axes
    fig, ax = plt.subplots()
    ax.boxplot(correlations)
    ax.set_xticklabels(["Siamese", "Triplet"])
    ax.set_ylim(-1, 1)
    ax.set_title("Correlations")
    ax.set_ylabel("Pearson's correlation coefficient")
    ax.set_xlabel("Architecture")
    # plt.show()
    plt.savefig(os.path.join(directory, "correlation_boxplot.png"))
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


def overlay_loss_rank(directory, trial_no):
    path = os.path.join(directory, str(trial_no))
    siamese = load_training_result(os.path.join(path, "siamese.pickle"))
    triplet = load_training_result(os.path.join(path, "triplet.pickle"))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(10, 4)

    loss_max = max(np.max(siamese.train_loss), np.max(triplet.train_loss))
    rank_max = max(np.max(siamese.train_rank), np.max(triplet.train_rank))

    loss_rank_overlay(siamese.train_loss, siamese.train_rank, ax1, "Siamese", loss_max, rank_max, siamese.pearson()[0])
    loss_rank_overlay(triplet.train_loss, triplet.train_rank, ax2, "Triplet", loss_max, rank_max, triplet.pearson()[0])

    fig.suptitle("Loss vs. Rank, Trial #{0}".format(trial_no))
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "loss_rank_overlay.png"), dpi=180)
    plt.close()


def wilcox_test(correlations):
    diff, p = wilcoxon(correlations[:, 1], correlations[:, 3])
    average_diff = diff / len(correlations)
    return average_diff, p
