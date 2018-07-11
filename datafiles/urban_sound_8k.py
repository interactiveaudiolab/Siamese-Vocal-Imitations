import csv
import os

from utils import preprocessing
from utils.utils import load_npy


def path_to_fold_n(audio_path):
    n = ''
    while audio_path[-1:].isdigit():
        n += audio_path[-1:]
        audio_path = audio_path[:-1]
    return int(n[::-1]) - 1  # folds are 1 indexed in path names


def calculate_spectrograms():
    """
    Calculates normalized imitation and reference spectrograms and saves them as .npy files.
    """
    data_dir = os.environ['SIAMESE_DATA_DIR']
    audio_path = os.path.join(data_dir, "UrbanSound8K", "audio")
    audio_paths = []
    for path in os.listdir(audio_path):
        abs_path = os.path.join(audio_path, path)
        if os.path.isdir(abs_path):
            audio_paths.append(abs_path)

    labels = []
    for i in range(10):
        labels.insert(i, {})

    csv_path = os.path.join(data_dir, 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    label_n = {}
    n = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold = int(row['fold']) - 1  # folds are 1 indexed in the CSV
            path = os.path.join(audio_path, 'fold{0}'.format(fold + 1), row['slice_file_name'])
            labels[fold][path] = row['class']
            if row['class'] not in label_n:
                label_n[row['class']] = n
                n += 1

    for audio_path in audio_paths:
        paths = preprocessing.recursive_wav_paths(audio_path)
        fold_n = path_to_fold_n(audio_path)
        preprocessing.calculate_spectrograms(paths, labels[fold_n], label_n, 'urbansound_{0}'.format(fold_n), preprocessing.reference_spectrogram)


class UrbanSound8K:
    def __init__(self, recalculate_spectrograms=False) -> None:
        super().__init__()
        if recalculate_spectrograms:
            calculate_spectrograms()

        self.folds = []
        self.fold_labels = []
        for fold_n in range(10):
            fold = load_npy('urbansound_{0}.npy'.format(fold_n))
            label = load_npy('urbansound_{0}_labels.npy'.format(fold_n))
            self.folds.append(fold)
            self.fold_labels.append(label)
