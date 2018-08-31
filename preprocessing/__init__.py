import audioop
import logging
import os
import pathlib

import audaugio
import librosa
import numpy as np

from data_files import Datafiles
from utils.progress_bar import Bar
from utils.utils import get_npy_dir


class Preprocessor:
    def __init__(self, imitation_augmentations: audaugio.ChainBase, reference_augmentations: audaugio.ChainBase):
        self.imitation_augmentations = imitation_augmentations
        self.reference_augmentations = reference_augmentations

    def __call__(self, datafiles: Datafiles):
        datafiles.prepare_spectrogram_calculation()

        bar = Bar("Calculating imitation spectrograms...", max=len(datafiles.imitation_paths))
        imitation_categories = set(datafiles.imitation_path_labels.values())
        self.imitation_file_indices = {label: 0 for label in imitation_categories}
        for path in datafiles.imitation_paths:
            self._calculate_imitation_spectrograms(path, datafiles.imitation_path_labels, datafiles.name, self.imitation_augmentations)
            bar.next()
        bar.finish()

        bar = Bar("Calculating reference spectrograms...", max=len(datafiles.reference_paths))
        reference_categories = set(l['label'] for l in datafiles.reference_path_labels.values())
        self.reference_file_indices = {label: 0 for label in reference_categories}
        for path in datafiles.reference_paths:
            self._calculate_reference_spectrograms(path, datafiles.reference_path_labels, datafiles.name, self.reference_augmentations)
            bar.next()
        bar.finish()

    def _calculate_reference_spectrograms(self, path, file_labels, dataset_name, augmentations):
        label = file_labels[path]
        spectrograms = reference_spectrogram(path, augmentations)
        spectrograms = self._normalize_spectrograms(np.array(spectrograms))
        j = 0
        i = self.reference_file_indices[label['label']]
        for spectrogram in spectrograms:
            if label['is_canonical']:
                file_name = 'canonical_reference.npy'
            else:
                i = self.reference_file_indices[label['label']]
                file_name = 'noncanonical_reference_{0}_{1}.npy'.format(i, j)
                j += 1
            self._save_npy(spectrogram, os.path.join('references', label['label'], file_name), dataset_name, "float32")

        self.reference_file_indices[label['label']] += 1

    def _calculate_imitation_spectrograms(self, path, file_labels, dataset_name, augmentations):
        label = file_labels[path]
        spectrograms = imitation_spectrogram(path, augmentations)
        spectrograms = self._normalize_spectrograms(np.array(spectrograms))
        i = self.imitation_file_indices[label]
        for j, spectrogram in enumerate(spectrograms):
            self._save_npy(spectrogram, os.path.join('imitations', label, 'imitation_{0}_{1}.npy'.format(i, j)), dataset_name, "float32")
        self.imitation_file_indices[label] += 1

    @staticmethod
    def _normalize_spectrograms(spectrograms):
        """
        data: (num_examples, num_freq_bins, num_time_frames)
        """
        if len(spectrograms) == 0:
            return []

        # filter out any all-0 spectrograms, TODO remove this one audaugio is modified to handle bypasses
        spectrograms = np.array([s for s in spectrograms if not s.min == s.max == 0])

        m = np.mean(spectrograms, axis=(1, 2))
        m = m.reshape(m.shape[0], 1, 1)
        std = np.std(spectrograms, axis=(1, 2))
        std = std.reshape(std.shape[0], 1, 1)

        m_matrix = np.repeat(np.repeat(m, spectrograms.shape[1], axis=1), spectrograms.shape[2], axis=2)
        std_matrix = np.repeat(np.repeat(std, spectrograms.shape[1], axis=1), spectrograms.shape[2], axis=2)
        normed = np.multiply(spectrograms - m_matrix, 1. / std_matrix)

        return normed

    @staticmethod
    def _save_npy(array, file_name, dataset, ar_type=None):
        array = np.array(array)
        if ar_type:
            array = array.astype(ar_type)
        path = os.path.join(get_npy_dir(dataset), file_name)
        try:
            np.save(path, array)
        except FileNotFoundError:  # can occur when the parent directory doesn't exist
            pathlib.Path(path).parent.mkdir(parents=True)
            np.save(path, array)


def imitation_spectrogram(path, augmentations: audaugio.ChainBase):
    """
    Calculate the spectrogram of an imitation located at path.

    Code written by Bongjun Kim.
    :param path: absolute path to the audio file
    :param augmentations:
    :return: power log spectrogram
    """
    try:
        y, sr = librosa.load(path, sr=16000)
    except audioop.error as e:
        logger = logging.getLogger('logger')
        logger.warning("Could not load {0}\n{1}".format(path, e))
        return None

    augmented_audio = augmentations(y, sr)

    spectrograms = []
    for audio in augmented_audio:
        # zero-padding
        if audio.shape[0] < 4 * sr:
            pad = np.zeros((4 * sr - audio.shape[0]))
            y_fix = np.append(audio, pad)
        else:
            y_fix = audio[0:int(4 * sr)]
        s = librosa.feature.melspectrogram(y=y_fix, sr=sr, n_fft=133, hop_length=133, power=2, n_mels=39, fmin=0.0, fmax=5000)
        s = s[:, :482]
        s = librosa.power_to_db(s, ref=np.max)
        spectrograms.append(s)
    return spectrograms


def reference_spectrogram(path, augmentations: audaugio.ChainBase):
    """
    Calculate the spectrogram of a reference recording located at path.

    Code written by Bongjun Kim.
    :param path: absolute path to the audio file
    :param augmentations:
    :return: power log spectrogram
    """
    try:
        y, sr = librosa.load(path, sr=44100)
    except audioop.error as e:
        logger = logging.getLogger('logger')
        logger.warning("Could not load {0}\n{1}".format(path, e))
        return None

    augmented_audio = augmentations(y, sr)

    spectrograms = []
    for audio in augmented_audio:
        if audio.shape[0] < 4 * sr:
            pad = np.zeros((4 * sr - audio.shape[0]))
            y_fix = np.append(audio, pad)
        else:
            y_fix = audio[0:int(4 * sr)]
        s = librosa.feature.melspectrogram(y=y_fix, sr=sr, n_fft=1024, hop_length=1024, power=2)
        s = librosa.power_to_db(s, ref=np.max)
        s = s[:, 0:128]
        spectrograms.append(s)
    return spectrograms