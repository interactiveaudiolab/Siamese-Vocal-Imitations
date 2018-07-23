import csv
import logging
import os

import numpy as np
from utils.progress_bar import Bar

import utils.preprocessing as preprocessing
from utils import utils
from utils.utils import zip_shuffle, get_dataset_dir


class VocalSketch:
    def __init__(self, train_ratio, val_ratio, test_ratio, version, shuffle=True, recalculate_spectrograms=False):
        if recalculate_spectrograms:
            self.calculate_spectrograms()

        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Training, validation, and testing ratios must add to 1")

        logger = logging.getLogger('logger')
        logger.info("train, validation, test ratios = {0}, {1}, {2}".format(train_ratio, val_ratio, test_ratio))

        try:
            references = utils.load_npy("references.npy", version)
            reference_labels = utils.load_npy("references_labels.npy", version)

            imitations = utils.load_npy("imitations.npy", version)
            imitation_labels = utils.load_npy("imitations_labels.npy", version)
        except FileNotFoundError:
            logger.warning("No saved spectrograms for Vocal Sketch (version: {0}). Calculating them...".format(version))
            self.calculate_spectrograms()

            references = utils.load_npy("references.npy", version)
            reference_labels = utils.load_npy("references_labels.npy", version)

            imitations = utils.load_npy("imitations.npy", version)
            imitation_labels = utils.load_npy("imitations_labels.npy", version)

        if shuffle:
            references, reference_labels = zip_shuffle(references, reference_labels)
            imitations, imitation_labels = zip_shuffle(imitations, imitation_labels)

        n_train_val = int(train_ratio * len(references) + val_ratio * len(references))
        n_test = int(test_ratio * len(references))

        # Split references up into training/validation set and testing set
        train_val_ref = references[:n_train_val]
        train_val_ref_labels = reference_labels[:n_train_val]
        test_ref = references[n_train_val:]
        test_ref_labels = reference_labels[n_train_val:]

        # Then, divide references over training and validation
        train_val_imit, train_val_imit_lab = self.filter_imitations(imitations, imitation_labels, train_val_ref_labels)
        try:
            n_train_imit = int(train_ratio / (train_ratio + val_ratio) * len(train_val_imit))
        except ZeroDivisionError:
            n_train_imit = 0
        train_imit = train_val_imit[:n_train_imit]
        val_imit = train_val_imit[n_train_imit:]
        train_imit_labels = train_val_imit_lab[:n_train_imit]
        val_imit_labels = train_val_imit_lab[n_train_imit:]

        self.train = VocalSketchPartition(train_val_ref, train_val_ref_labels, train_imit, train_imit_labels, "training")
        self.val = VocalSketchPartition(train_val_ref, train_val_ref_labels, val_imit, val_imit_labels, "validation")
        self.test = VocalSketchPartition(test_ref, test_ref_labels, imitations, imitation_labels, "testing")

    @staticmethod
    def calculate_spectrograms():
        raise NotImplementedError

    @staticmethod
    def filter_imitations(all_imitations, all_imitation_labels, reference_labels):
        imitations = []
        imitation_labels = []
        for i, l in zip(all_imitations, all_imitation_labels):
            if l in reference_labels:
                imitations.append(i)
                imitation_labels.append(l)

        return imitations, imitation_labels


class VocalSketch_v1(VocalSketch):
    def __init__(self, train_ratio, val_ratio, test_ratio, shuffle=True, recalculate_spectrograms=False):
        super().__init__(train_ratio, val_ratio, test_ratio, "vs1.0", shuffle, recalculate_spectrograms)

    @staticmethod
    def calculate_spectrograms():
        """
        Calculates normalized imitation and reference spectrograms and saves them as .npy files.
        """
        data_dir = get_dataset_dir()
        imitation_path = os.path.join(data_dir, "vs1.0/vocal_imitations/included")
        reference_path = os.path.join(data_dir, "vs1.0/sound_recordings")

        imitation_paths = preprocessing.recursive_wav_paths(imitation_path)
        reference_paths = preprocessing.recursive_wav_paths(reference_path)

        reference_csv = os.path.join(data_dir, 'vs1.0', "sound_recordings.csv")
        imitation_csv = os.path.join(data_dir, 'vs1.0', "vocal_imitations.csv")

        reference_labels = {}
        with open(reference_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(reference_path, row['filename'])
                reference_labels[path] = row['sound_label']

        imitation_labels = {}
        with open(imitation_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(imitation_path, row['filename'])
                imitation_labels[path] = row['sound_label']

        n = 0
        label_no = {}
        for file_name, label in reference_labels.items():
            if label not in label_no:
                label_no[label] = n
                n += 1

        preprocessing.calculate_spectrograms(imitation_paths, imitation_labels, label_no, 'imitations', 'vs1.0', preprocessing.imitation_spectrogram)
        preprocessing.calculate_spectrograms(reference_paths, reference_labels, label_no, 'references', 'vs1.0', preprocessing.reference_spectrogram)


class VocalSketch_v2(VocalSketch):
    def __init__(self, train_ratio, val_ratio, test_ratio, shuffle=True, recalculate_spectrograms=False):
        super().__init__(train_ratio, val_ratio, test_ratio, "vs2.0", shuffle, recalculate_spectrograms)

    @staticmethod
    def calculate_spectrograms():
        """
        Calculates normalized imitation and reference spectrograms and saves them as .npy files.
        """
        data_dir = get_dataset_dir()
        imitation_path_1 = os.path.join(data_dir, "vs2.0/vocal_imitations/included")
        imitation_path_2 = os.path.join(data_dir, "vs2.0/vocal_imitations_set2/included")
        reference_path = os.path.join(data_dir, "vs2.0/sound_recordings")

        imitation_paths = preprocessing.recursive_wav_paths(imitation_path_1) + preprocessing.recursive_wav_paths(imitation_path_2)
        reference_paths = preprocessing.recursive_wav_paths(reference_path)

        imitation_csv_1 = os.path.join(data_dir, 'vs2.0', "vocal_imitations.csv")
        imitation_csv_2 = os.path.join(data_dir, 'vs2.0', "vocal_imitaitons_set2.csv")  # not a typo, the CSV's name is misspelled
        reference_csv = os.path.join(data_dir, 'vs2.0', "sound_recordings.csv")

        reference_labels = {}
        with open(reference_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(reference_path, row['filename'])
                reference_labels[path] = row['sound_label']

        imitation_labels = {}
        with open(imitation_csv_1) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(imitation_path_1, row['filename'])
                imitation_labels[path] = row['sound_label']

        with open(imitation_csv_2) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(imitation_path_2, row['filename'])
                imitation_labels[path] = row['sound_label']

        n = 0
        label_no = {}
        for file_name, label in reference_labels.items():
            if label not in label_no:
                label_no[label] = n
                n += 1

        preprocessing.calculate_spectrograms(imitation_paths, imitation_labels, label_no, 'imitations', 'vs2.0', preprocessing.imitation_spectrogram)
        preprocessing.calculate_spectrograms(reference_paths, reference_labels, label_no, 'references', 'vs2.0', preprocessing.reference_spectrogram)


class VocalSketchPartition:
    def __init__(self, references, reference_labels, all_imitations, all_imitation_labels, dataset_type):
        self.references = references
        self.reference_labels = reference_labels

        # filter out imitations of things that are not in this set at all
        self.imitations = []
        self.imitation_labels = []
        for i, l in zip(all_imitations, all_imitation_labels):
            if l in reference_labels:
                self.imitations.append(i)
                self.imitation_labels.append(l)

        bar = Bar("Creating pairs for {0}...".format(dataset_type), max=len(self.references) * len(self.imitations))
        self.positive_pairs = []
        self.negative_pairs = []
        self.all_pairs = []
        self.labels = np.zeros([len(self.imitations), len(self.references)])
        for i, (imitation, imitation_label) in enumerate(zip(self.imitations, self.imitation_labels)):
            for j, (reference, reference_label) in enumerate(zip(self.references, self.reference_labels)):
                if reference_label == imitation_label:
                    self.positive_pairs.append([imitation, reference, True])
                    self.all_pairs.append([imitation, reference, True])
                    self.labels[i, j] = 1
                else:
                    self.negative_pairs.append([imitation, reference, False])
                    self.all_pairs.append([imitation, reference, False])
                    self.labels[i, j] = 0

                bar.next()
        bar.finish()
