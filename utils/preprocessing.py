import csv
import os

import librosa
import numpy as np

from progress.bar import Bar

from utils.utils import save_npy


def load_data_set():
    """
    Calculates normalized imitation and reference spectrograms and saves them as .npy files.
    """
    data_dir = os.environ['SIAMESE_DATA_DIR']
    imitation_paths = recursive_wav_paths(data_dir + "/vocal_imitations/included")
    reference_paths = recursive_wav_paths(data_dir + "/sound_recordings")
    reference_csv = os.path.join(data_dir, "sound_recordings.csv")
    imitation_csv = os.path.join(data_dir, "vocal_imitations.csv")

    reference_labels = {}
    with open(reference_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            reference_labels[row['filename']] = row['sound_label']

    imitation_labels = {}
    with open(imitation_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            imitation_labels[row['filename']] = row['sound_label']

    n = 0
    label_no = {}
    for file_name, label in reference_labels.items():
        if label not in label_no:
            label_no[label] = n
            n += 1

    calculate_spectrograms(imitation_paths, imitation_labels, label_no, 'imitation', imitation_spectrogram)
    calculate_spectrograms(reference_paths, reference_labels, label_no, 'reference', reference_spectrogram)


def calculate_spectrograms(paths, file_labels, label_no, file_type, spectrogram_func):
    # calculate spectrograms and save
    bar = Bar('Calculating {0} spectrograms...'.format(file_type), max=len(paths))
    spectrograms = []
    labels = []
    for path in paths:
        spectrogram = spectrogram_func(path)
        spectrograms.append(spectrogram)

        file_name = os.path.basename(path)
        label = file_labels[file_name]
        labels.append(label_no[label])

        bar.next()

    spectrograms = normalize_spectrograms(np.array(spectrograms))
    save_npy(spectrograms, '{0}s.npy'.format(file_type), "float32")
    save_npy(labels, '{0}_labels.npy'.format(file_type), "uint8")
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


def reference_spectrogram(path):
    """
    Calculate the spectrogram of a reference recording located at path.

    Code written by Bongjun Kim.
    :param path: absolute path to the audio file
    :return: power log spectrogram
    """
    y, sr = librosa.load(path, sr=44100)
    # zero-padding
    if y.shape[0] < 4 * sr:
        pad = np.zeros((4 * sr - y.shape[0]))
        y_fix = np.append(y, pad)
    else:
        y_fix = y[0:int(4 * sr)]
    S = librosa.feature.melspectrogram(y=y_fix, sr=sr, n_fft=1024, hop_length=1024, power=2)
    S = librosa.power_to_db(S, ref=np.max)
    S_fix = S[:, 0:128]
    return S_fix


def imitation_spectrogram(path):
    """
    Calculate the spectrogram of an imitation located at path.

    Code written by Bongjun Kim.
    :param path: absolute path to the audio file
    :return: power log spectrogram
    """
    y, sr = librosa.load(path, sr=16000)
    # zero-padding
    if y.shape[0] < 4 * sr:
        pad = np.zeros((4 * sr - y.shape[0]))
        y_fix = np.append(y, pad)
    else:
        y_fix = y[0:int(4 * sr)]
    S = librosa.feature.melspectrogram(y=y_fix, sr=sr, n_fft=133,
                                       hop_length=133, power=2, n_mels=39,
                                       fmin=0.0, fmax=5000)
    S = S[:, :482]
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def normalize_spectrograms(spectrograms):
    """
    data: (num_examples, num_freq_bins, num_time_frames)
    """

    m = np.mean(spectrograms, axis=(1, 2))
    m = m.reshape(m.shape[0], 1, 1)
    std = np.std(spectrograms, axis=(1, 2))
    std = std.reshape(std.shape[0], 1, 1)

    m_matrix = np.repeat(np.repeat(m, spectrograms.shape[1], axis=1), spectrograms.shape[2], axis=2)
    std_matrix = np.repeat(np.repeat(std, spectrograms.shape[1], axis=1), spectrograms.shape[2], axis=2)

    normed = np.multiply(spectrograms - m_matrix, 1. / std_matrix)

    return normed
