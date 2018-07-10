import logging
import os

import numpy as np
from utils.progress_bar import Bar

from utils import preprocessing
from utils.utils import load_npy


def path_to_fold_n(audio_path):
    n = ''
    while audio_path[-1:].isdigit():
        n += audio_path[-1:]
        audio_path = audio_path[:-1]
    return int(n[::-1]) - 1  # folds are 1 indexed in path names


def calculate_spectrograms(per_language, n_batches):
    """
    Calculates normalized imitation and reference spectrograms and saves them as .npy files.
    :param n_batches:
    """
    logger = logging.getLogger('logger')
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
    batch_size = int(len(all_paths) / n_batches)
    logger.debug("Calculating voxforge spectrograms in batches of size {0}...".format(batch_size))
    for batch_n in range(n_batches):
        logger.debug("Batch {0} of {1}...".format(batch_n, n_batches))
        batch = all_paths[batch_size * batch_n: batch_size * (batch_n + 1)]
        preprocessing.calculate_spectrograms(batch, labels, label_n, 'voxforge_{0}'.format(batch_n), preprocessing.imitation_spectrogram)


class Voxforge:
    def __init__(self, recalculate_spectrograms=False) -> None:
        super().__init__()
        self.per_language = 8000
        n_batches = 50
        if recalculate_spectrograms:
            calculate_spectrograms(self.per_language, n_batches)

        self.audio = np.empty(shape=[0, 39, 482])
        self.labels = np.array([])
        bar = Bar("Loading voxforge spectrograms from file", max=n_batches)
        for batch_n in range(n_batches):
            audio_batch = load_npy('voxforge_{0}.npy'.format(batch_n))
            self.audio = np.concatenate([self.audio, audio_batch])
            label_batch = load_npy('voxforge_{0}_labels.npy'.format(batch_n))
            self.labels = np.concatenate([self.labels, label_batch])
            bar.next()
        bar.finish()

        self.unique_labels = []
        for l in self.labels:
            if l not in self.unique_labels:
                self.unique_labels.append(l)
