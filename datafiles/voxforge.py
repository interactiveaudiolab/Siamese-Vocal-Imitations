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
    sub_path = os.path.join(data_dir, "voxforge")
    sub_paths = []
    all_labels = []
    for path in os.listdir(sub_path):
        abs_path = os.path.join(sub_path, path)
        if os.path.isdir(abs_path):
            sub_paths.append(abs_path)
            all_labels.append(path)

    label_n = {}
    n = 0
    for label in all_labels:
        if label not in label_n:
            label_n[label] = n
            n += 1

    all_paths = []
    labels = {}
    for sub_path, label in zip(sub_paths, all_labels):
        paths = preprocessing.recursive_wav_paths(sub_path)
        for path in paths:
            all_paths.append(path)
            labels[os.path.basename(path)] = label
    preprocessing.calculate_spectrograms(all_paths, labels, label_n, 'voxforge', preprocessing.imitation_spectrogram)


class Voxforge:
    def __init__(self, recalculate_spectrograms=False) -> None:
        super().__init__()
        if recalculate_spectrograms:
            calculate_spectrograms()

        self.audio = load_npy('voxforge.npy')
        self.labels = load_npy('voxforge_labels.npy')
        self.unique_labels = []
        for l in self.labels:
            if l not in self.unique_labels:
                self.unique_labels.append(l)
