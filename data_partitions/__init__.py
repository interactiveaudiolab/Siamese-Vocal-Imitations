import logging
import os
import pickle
from typing import List

import numpy as np

from data_partitions.split import PartitionSplit
from utils.utils import get_npy_dir


class Spectrogram:
    def __init__(self, path):
        self.path = path

    def load(self):
        return np.load(self.path)


class Imitation(Spectrogram):
    def __init__(self, path: str, label: str, index: int):
        super().__init__(path)
        self.label = label
        self.index = index


class Reference(Spectrogram):
    def __init__(self, path: str, label: str, is_canonical: bool, index: int):
        super().__init__(path)
        self.label = label
        self.is_canonical = is_canonical
        self.index = index


class PairPartition:
    def __init__(self, imitations: List[Imitation], references: List[Reference]):
        self.imitations = imitations
        self.references = references

        self.positive = []
        self.negative_coarse = []
        self.negative_fine = []
        self.all_pairs = []
        self.labels = np.zeros(shape=[len(imitations), len(references)])

        for i, imitation in enumerate(self.imitations):
            for j, reference in enumerate(self.references):
                label = self._classify_pair(imitation, reference)
                self.all_pairs.append([imitation, reference, label])
                self.labels[i, j] = label

    def _classify_pair(self, imitation, reference):
        if reference.label == imitation.label:
            if reference.is_canonical:
                label = True
                self.positive.append([imitation, reference, label])
            else:
                label = False
                self.negative_fine.append([imitation, reference, label])
        else:
            label = False
            self.negative_coarse.append([imitation, reference, label])
        return label


class TripletPartition:
    def __init__(self, imitations: List[Imitation], references: List[Reference]):
        self.imitations = imitations
        self.references = references

        self.positive_fine = []
        self.positive_coarse = []
        self.negative_fine = []
        self.negative_coarse = []

        for i, imitation in enumerate(self.imitations):
            for j, near in enumerate(self.references):
                for k, far in enumerate(self.references):
                    self._classify_triplet(imitation, far, near)

    def _classify_triplet(self, imitation, far, near):
        if near.label == imitation.label and near.is_canonical:
            label = True
            if far.label == imitation.label:
                self.positive_fine.append([imitation, near, far, label])
            else:
                self.positive_coarse.append([imitation, near, far, label])
        elif far.label == imitation.label and far.is_canonical:
            label = False
            if near.label == imitation.label:
                self.negative_fine.append([imitation, near, far, label])
            else:
                self.negative_coarse.append([imitation, near, far, label])
        else:
            label = None
        return label


class Partitions:
    def __init__(self, name, split: PartitionSplit):
        self.name = name
        self.data_root = get_npy_dir(self.name)

        all_categories = self._get_all_categories()
        np.random.shuffle(all_categories)
        # split the categories between train/val and test
        train_val_categories, test_categories = split.split_categories(all_categories)
        # split the imitations between train and val
        train_imitations, val_imitations = self._split_train_val_imitations(train_val_categories, split)
        # get the remaining imitations that belong to test
        test_imitations = self._get_test_imitations(test_categories)

        self.train_args = (train_imitations, self._categories_to_references(train_val_categories))
        self.val_args = (val_imitations, self._categories_to_references(train_val_categories))
        self.test_args = (test_imitations, self._categories_to_references(test_categories))

        self.train = None
        self.val = None
        self.test = None

    def generate_partitions(self, partition_type):
        self.train = partition_type(*self.train_args)
        self.val = partition_type(*self.val_args)
        self.test = partition_type(*self.test_args)

    def save(self, location):
        logger = logging.getLogger('logger')

        logger.debug("Saving partitions at {0}...".format(location))
        with open(location, 'wb') as f:
            pickle.dump(self.train_args, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.val_args, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.test_args, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, location):
        logger = logging.getLogger('logger')

        try:
            logger.debug("Loading partitions from {0}...".format(location))

            with open(location, 'rb') as f:
                self.train_args = pickle.load(f)
                self.val_args = pickle.load(f)
                self.test_args = pickle.load(f)
        except FileNotFoundError as e:
            logger.critical("No pickled partition at {0}".format(location))
            raise e
        except EOFError as e:
            logger.critical("Insufficient amount of data found in {0}".format(location))
            raise e

    def _get_all_categories(self):
        imitation_path = os.path.join(self.data_root, 'imitations')
        imitation_categories = set(next(os.walk(imitation_path))[1])

        reference_path = os.path.join(self.data_root, 'references')
        reference_categories = set(next(os.walk(reference_path))[1])

        if imitation_categories != reference_categories:
            missing_in_imitation = reference_categories - imitation_categories
            missing_in_reference = imitation_categories - reference_categories
            logger = logging.getLogger('logger')
            missing_message = "{0} categories were found in the {1} set but not in the {2} set. Ignoring them."
            if missing_in_imitation:
                logger.info(missing_message.format(len(missing_in_imitation), 'reference', 'imitation'))
            if missing_in_reference:
                logger.info(missing_message.format(len(missing_in_reference), 'imitation', 'reference'))

        return list(reference_categories.intersection(imitation_categories))

    def _split_train_val_imitations(self, categories, split):
        train = []
        val = []
        for category in categories:
            path = os.path.join(self.data_root, 'imitations', category)
            imitation_paths = [os.path.join(path, p) for p in os.listdir(path)]
            np.random.shuffle(imitation_paths)
            imitations = []
            for imitation_path in imitation_paths:
                imitations.append(Imitation(imitation_path, category, int(os.path.basename(imitation_path).split('_')[1])))
            this_train, this_val = split.split_imitations(imitations)
            train += this_train
            val += this_val
        return train, val

    def _get_test_imitations(self, categories):
        imitations = []
        for category in categories:
            path = os.path.join(self.data_root, 'imitations', category)
            imitations += os.listdir(path)
        return imitations

    def _categories_to_references(self, categories):
        references = []
        for category in categories:
            path = os.path.join(self.data_root, 'references', category)
            reference_paths = [os.path.join(path, p) for p in os.listdir(path)]
            for reference_path in reference_paths:
                is_canonical = os.path.basename(reference_path) == 'canonical_reference.npy'
                index = -1 if is_canonical else int(os.path.basename(reference_path).split('_')[2])
                references.append(Reference(reference_path, category, is_canonical, index))
        return references


if __name__ == '__main__':
    _split = PartitionSplit(.3, .2, .5)
    Partitions('vocal_imitation', _split)
