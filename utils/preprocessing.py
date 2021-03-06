import audioop
import logging
import os
import pathlib

import audaugio
import librosa
import numpy as np

from utils.progress_bar import Bar
from utils.utils import get_npy_dir


def calculate_spectrograms(paths, file_labels, file_name, dataset_name, spectrogram_func, augmentations):
    bar = Bar('Calculating spectrograms and saving them at {0}.npy...'.format(os.path.join(get_npy_dir(dataset_name), file_name)), max=len(paths))
    all_spectrograms = []
    labels = []
    for path in paths:
        spectrograms = spectrogram_func(path, augmentations)
        for spectrogram in spectrograms:
            all_spectrograms.append(spectrogram)
            label = file_labels[path]
            labels.append(label)

        bar.next()

    all_spectrograms = normalize_spectrograms(np.array(all_spectrograms))
    save_npy(all_spectrograms, '{0}.npy'.format(file_name), dataset_name, "float32")
    save_npy(labels, '{0}_labels.npy'.format(file_name), dataset_name)
    bar.finish()


def recursive_wav_paths(path):
    """
    Get the paths of all .wav files found recursively in the path.

    :param path: path to search recursively in
    :return: list of absolute paths
    """
    absolute_paths = []
    for folder, subs, files in os.walk(path):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension.lower() == '.wav':
                file_path = os.path.join(folder, file)
                absolute_paths.append(os.path.abspath(file_path))

    return absolute_paths


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
        s = librosa.feature.melspectrogram(y=y_fix, sr=sr, n_fft=133,
                                           hop_length=133, power=2, n_mels=39,
                                           fmin=0.0, fmax=5000)
        s = s[:, :482]
        s = librosa.power_to_db(s, ref=np.max)
        spectrograms.append(s)
    return spectrograms


def normalize_spectrograms(spectrograms):
    """
    data: (num_examples, num_freq_bins, num_time_frames)
    """
    if len(spectrograms) == 0:
        return []

    m = np.mean(spectrograms, axis=(1, 2))
    m = m.reshape(m.shape[0], 1, 1)
    std = np.std(spectrograms, axis=(1, 2))
    std = std.reshape(std.shape[0], 1, 1)

    m_matrix = np.repeat(np.repeat(m, spectrograms.shape[1], axis=1), spectrograms.shape[2], axis=2)
    std_matrix = np.repeat(np.repeat(std, spectrograms.shape[1], axis=1), spectrograms.shape[2], axis=2)

    normed = np.multiply(spectrograms - m_matrix, 1. / std_matrix)

    return normed


def save_npy(array, file_name, dataset, ar_type=None):
    array = np.array(array)
    if ar_type:
        array = array.astype(ar_type)
    path = os.path.join(get_npy_dir(dataset), file_name)
    try:
        np.save(path, array)
    except FileNotFoundError:  # can occur when the parent directory doesn't exist
        pathlib.Path(path).mkdir(parents=True)
        np.save(path, array)