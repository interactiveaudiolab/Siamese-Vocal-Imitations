import os

import numpy as np
from progress.bar import Bar

from utils import preprocessing
from utils.utils import load_npy


def path_to_fold_n(audio_path):
    n = ''
    while audio_path[-1:].isdigit():
        n += audio_path[-1:]
        audio_path = audio_path[:-1]
    return int(n[::-1]) - 1  # folds are 1 indexed in path names


def calculate_spectrograms(per_language):
    """
    Calculates normalized imitation and reference spectrograms and saves them as .npy files.
    """
    data_dir = os.path.join(os.environ['SIAMESE_DATA_DIR'], "voxforge")
    languages = []
    for path in os.listdir(data_dir):
        abs_path = os.path.join(data_dir, path)
        if os.path.isdir(abs_path):
            languages.append(path)

    label_n = {}
    n = 0
    for language in languages:
        if language not in label_n:
            label_n[language] = n
            n += 1

    all_paths = []
    labels = {}
    bar = Bar("Preparing to calculate voxforge spectrograms...", max=len(languages) * per_language)
    for language in languages:
        language_dir = os.path.join(data_dir, language)
        paths = preprocessing.recursive_wav_paths(language_dir)
        np.random.shuffle(paths)
        paths = paths[:per_language]
        for path in paths:
            all_paths.append(path)
            labels[os.path.basename(path)] = language
            bar.next()
    bar.finish()
    preprocessing.calculate_spectrograms(all_paths, labels, label_n, 'voxforge', preprocessing.imitation_spectrogram)


class Voxforge:
    def __init__(self, recalculate_spectrograms=False) -> None:
        super().__init__()
        self.per_language = 8000
        if recalculate_spectrograms:
            calculate_spectrograms(self.per_language)

        self.audio = load_npy('voxforge.npy')
        self.labels = load_npy('voxforge_labels.npy')
        self.unique_labels = []
        for l in self.labels:
            if l not in self.unique_labels:
                self.unique_labels.append(l)
