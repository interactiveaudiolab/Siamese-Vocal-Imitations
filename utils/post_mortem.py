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
    try:
        progress.load(path)
    except FileNotFoundError:
        return None
    except EOFError:
        with open(path, 'rb') as f:
            progress = pickle.load(f)
            progress.save(path)
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
    utils.graphing.boxplot(directory, correlations[:, [1, 3]], p_value)


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
    print(diff, p)
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
