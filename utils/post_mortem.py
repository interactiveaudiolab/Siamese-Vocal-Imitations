import csv
import os
import pickle
import shutil
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_training_result(path):
    try:
        with open(path, 'rb') as f:
            result = pickle.load(f)
            return result
    except FileNotFoundError:
        print("{0} not found, skipping".format(path))


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
    return correlations


def generate_correlation_csv(directory, correlations):
    csv_path = os.path.join(directory, "correlations.csv")
    with open(csv_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_no', 'siamese_tr', 'siamese_vl', 'triplet_tr', 'triplet_vl'])
        for b in correlations:
            writer.writerow(b)


def generate_boxplot(directory, correlations):
    correlations = np.array(correlations)
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


def condense_graphs(directory):
    for file in glob(os.path.join(directory, "*/*")):
        if file.endswith("png"):
            path = Path(file)
            dest_name = path.parents[0].name + "_" + path.name
            p = os.path.join(path.parents[1], dest_name)
            print("{0} --> {1}".format(path, p))
            shutil.copyfile(path, p)
