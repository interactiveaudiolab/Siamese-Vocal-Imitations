import csv
import os
import pickle
import shutil
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


def load_training_result(path):
    try:
        with open(path, 'rb') as f:
            result = pickle.load(f)
            return result
    except FileNotFoundError:
        print("{0} not found, skipping".format(path))


def get_correlations(output_dir):
    correlations = []
    join = os.path.join(output_dir, "*/")
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


def generate_correlation_csv(output_dir, correlations):
    csv_path = os.path.join(output_dir, "correlations.csv")
    with open(csv_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_no', 'siamese_tr', 'siamese_vl', 'triplet_tr', 'triplet_vl'])
        for b in correlations:
            writer.writerow(b)


def generate_boxplot(output_dir, correlations):
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
    plt.savefig(os.path.join(output_dir, "correlation_boxplot.png"))
    plt.close()


def condense_graphs(directory):
    for roots, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("png"):
                src = os.path.join(roots, file)
                p = os.path.join(os.path.dirname(roots), os.path.basename(roots) + '_' + file)
                print("{0} --> {1}".format(src, p))
                shutil.copyfile(src, p)
