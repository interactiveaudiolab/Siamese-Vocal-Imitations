import csv
import os
import pickle
import shutil
from glob import glob


def load_training_result(path):
    try:
        with open(path, 'rb') as f:
            result = pickle.load(f)
            return result
    except FileNotFoundError:
        print("{0} not found, skipping".format(path))


def generate_correlation_csv(output_dir):
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

    csv_path = os.path.join(output_dir, "correlations.csv")
    with open(csv_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_no', 'siamese_tr', 'siamese_vl', 'triplet_tr', 'triplet_vl'])
        for b in correlations:
            writer.writerow(b)


def condense_graphs(directory):
    for roots, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("png"):
                src = os.path.join(roots, file)
                p = os.path.join(os.path.dirname(roots), os.path.basename(roots) + '_' + file)
                print("{0} --> {1}".format(src, p))
                shutil.copyfile(src, p)
