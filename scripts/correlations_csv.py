import csv
import os
import pickle
import sys
from glob import glob


def get_result(path):
    try:
        with open(path, 'rb') as f:
            result = pickle.load(f)
            return result
    except FileNotFoundError:
        print("{0} not found, skipping".format(path))


def main():
    print(os.getcwd())
    if len(sys.argv) <= 1:
        raise ValueError("python correlations_csv.py output_dir")

    output_dir = sys.argv[1]
    a = []
    for folder in glob(output_dir + "*/"):
        trial = int(os.path.basename(os.path.dirname(folder)))
        print("{0}\n".format(str(trial)))
        siamese_path = os.path.join(folder, 'siamese.pickle')
        siamese_result = get_result(siamese_path)

        triplet_path = os.path.join(folder, 'triplet.pickle')
        triplet_result = get_result(triplet_path)

        siamese_tr, siamese_vl = siamese_result.pearson()
        triplet_tr, triplet_vl = triplet_result.pearson()
        a.append([trial, siamese_tr, siamese_vl, triplet_tr, triplet_vl])

        print(siamese_result)
        print(triplet_result)

    with open("correlations.csv", 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_no', 'siamese_tr', 'siamese_vl', 'triplet_tr', 'triplet_vl'])
        for b in a:
            writer.writerow(b)


if __name__ == '__main__':
    main()
