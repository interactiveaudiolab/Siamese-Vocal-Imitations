import csv
import os
import pickle
import sys


def get_result(path):
    try:
        with open(path, 'rb') as f:
            result = pickle.load(f)
            return result
    except FileNotFoundError:
        print("{0} not found, skipping".format(path))


def main():
    print(os.getcwd())
    if len(sys.argv) < 2:
        raise ValueError("python correlations_csv.py start_trial end_trial")

    try:
        start_trial = int(sys.argv[1])
        end_trial = int(sys.argv[2])
    except ValueError:
        start_trial = sys.argv[1]
        end_trial = sys.argv[2]
        raise ValueError("{0} or {1} is not an integer".format(start_trial, end_trial))

    if end_trial < start_trial:
        raise ValueError("End trial must be >= start trial")

    output_dir = './output/{0}'
    a = []
    for trial in range(start_trial, end_trial + 1):
        print("{0}\n".format(str(trial)))
        current_dir = output_dir.format(trial)
        siamese_path = os.path.join(current_dir, 'siamese.pickle')
        siamese_result = get_result(siamese_path)

        bisiamese_path = os.path.join(current_dir, 'bisiamese.pickle')
        bisiamese_result = get_result(bisiamese_path)

        siamese_tr, siamese_vl = siamese_result.pearson()
        bisiamese_tr, bisiamese_vl = bisiamese_result.pearson()
        a.append([trial, siamese_tr, siamese_vl, bisiamese_tr, bisiamese_vl])

        print(siamese_result)
        print(bisiamese_result)

    with open("correlations.csv", 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_no', 'siamese_tr', 'siamese_vl', 'bisiamese_tr', 'bisiamese_vl'])
        for b in a:
            writer.writerow(b)


if __name__ == '__main__':
    main()
