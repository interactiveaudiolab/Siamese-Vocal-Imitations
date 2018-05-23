import os

import librosa
import numpy as np

from progress.bar import Bar


def preprocess():
    imitation_paths = get_audio_paths("./data/audio/imitation")
    reference_paths = get_audio_paths("./data/audio/reference")

    # build up dictionary of reference names -> reference #s
    label_n = 0
    label_dict = {}
    for reference_path in reference_paths:
        label_name = os.path.basename(reference_path)[:-4].lower()
        label_dict[label_name] = label_n
        label_n += 1

    # calculate imitation spectrograms and save
    imitation_bar = Bar("Calculating imitation spectrograms", max=len(imitation_paths))
    imitations = []
    imitation_labels = []
    for imitation_path in imitation_paths:
        spectrogram = get_imitation_spectrogram(imitation_path)

        label_name = os.path.basename(imitation_path)
        label_name = label_name[:label_name.find(" - ")].replace(" ", "_").replace("(", "").replace(")", "").lower()
        if label_name in label_dict:
            imitation_labels.append(label_dict[label_name])
            # imitations.append(spectrogram)
        else:
            print("Could not find imitation labeled {0} in dictionary.".format(label_name))
        imitation_bar.next()

    save(imitations, "imitations.npy", "float32")
    save(imitation_labels, "imitation_labels.npy", "uint8")

    imitation_bar.finish()

    # calculate reference spectrograms and save
    reference_bar = Bar("Calculating reference spectrograms", max=len(reference_paths))
    references = []
    reference_labels = []
    for reference_path in reference_paths:
        spectrogram = get_reference_spectrogram(reference_path)

        label_name = os.path.basename(reference_path)[:-4].lower()
        if label_name in label_dict:
            reference_labels.append(label_dict[label_name])
            # references.append(spectrogram)
        else:
            print("Could not find reference labeled {0} in dictionary.".format(label_name))
        reference_bar.next()

    save(references, "references.npy", "float32")
    save(reference_labels, "reference_labels.npy", "uint8")

    reference_bar.finish()


def save(array, suffix, type):
    np.save("./data/npy/" + suffix, np.array(array).astype(type))


def get_audio_paths(path):
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


def get_reference_spectrogram(path):
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


def get_imitation_spectrogram(path):
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
