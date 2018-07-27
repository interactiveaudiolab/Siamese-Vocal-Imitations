import logging
import pickle
import sys

import numpy as np

from data_files.generics import Datafiles
from utils.obj import DataSplit
from utils.progress_bar import Bar


class SiamesePartitions:
    def __init__(self, dataset: Datafiles, split: DataSplit, regenerate_splits=False):
        dataset_name = type(dataset).__name__
        pickle_name = "./partition_pickles/{0}.pickle".format(dataset_name)
        logger = logging.getLogger('logger')

        if regenerate_splits:
            logger.info("train, validation, test ratios = {0}, {1}, {2}".format(split.train_ratio, split.validation_ratio, split.test_ratio))

            imitations = dataset.imitations
            imitation_labels = dataset.imitation_labels
            references = dataset.references
            reference_labels = dataset.reference_labels

            # Split categories across train/val and test
            # i.e. train/val share a set of categories and test uses the other ones
            categories = list(set([v['label'] for v in reference_labels]))
            np.random.shuffle(categories)

            n_train_val = int(split.train_ratio * len(categories) + split.validation_ratio * len(categories))

            train_val_ref, train_val_ref_labels = self.split_references(references, reference_labels, categories[:n_train_val])
            test_ref, test_ref_labels = self.split_references(references, reference_labels, categories[n_train_val:])

            train_val_imit, train_val_imit_lab = self.filter_imitations(imitations, imitation_labels, train_val_ref_labels)

            train_imit, train_imit_lab, val_imit, val_imit_lab = self.split_imitations(categories, imitations, split, train_val_imit, train_val_imit_lab)

            self.train = SiamesePartition(train_val_ref, train_val_ref_labels, train_imit, train_imit_lab, "training")
            self.val = SiamesePartition(train_val_ref, train_val_ref_labels, val_imit, val_imit_lab, "validation")
            self.test = SiamesePartition(test_ref, test_ref_labels, imitations, imitation_labels, "testing")

            logger.debug("Saving partitions at {0}...".format(pickle_name))
            with open(pickle_name, 'wb') as f:
                pickle.dump(self.train, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.val, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.test, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            try:
                logger.debug("Loading partitions from {0}...".format(pickle_name))
                with open(pickle_name, 'rb') as f:
                    self.train = pickle.load(f)
                    self.val = pickle.load(f)
                    self.test = pickle.load(f)
            except FileNotFoundError:
                with open(pickle_name, 'w+b'):
                    logger.critical("No pickled partition at {0}".format(pickle_name))
                    sys.exit()
            except EOFError:
                logger.critical("Insufficient amount of data found in {0}".format(pickle_name))
                sys.exit()

    @staticmethod
    def split_imitations(categories, imitations, split, train_val_imit, train_val_imit_labels):
        imitation_shape = list(imitations.shape)
        imitation_shape[0] = 0
        train_imit = np.empty(imitation_shape)
        val_imit = np.empty(imitation_shape)
        train_imit_labels = np.empty(0)
        val_imit_labels = np.empty(0)
        for category in categories:
            ind = [i for i, v in enumerate(train_val_imit_labels) if v == category]
            np.random.shuffle(ind)
            n_train = int(split.train_ratio / (split.train_ratio + split.validation_ratio) * len(ind))

            imit = train_val_imit[ind]
            imit_labels = train_val_imit_labels[ind]

            train_imit = np.concatenate([train_imit, imit[:n_train]])
            train_imit_labels = np.concatenate([train_imit_labels, imit_labels[:n_train]])
            val_imit = np.concatenate([val_imit, imit[n_train:]])
            val_imit_labels = np.concatenate([val_imit_labels, imit_labels[n_train:]])
        return train_imit, train_imit_labels, val_imit, val_imit_labels

    @staticmethod
    def split_references(references, reference_labels, categories):
        ref = []
        ref_labels = []
        for r, l in zip(references, reference_labels):
            if l['label'] in categories:
                ref.append(r)
                ref_labels.append(l)
        return np.array(ref), np.array(ref_labels)

    @staticmethod
    def filter_imitations(all_imitations, all_imitation_labels, reference_labels):
        imitations = []
        imitation_labels = []
        reference_label_list = [v['label'] for v in reference_labels]
        for i, l in zip(all_imitations, all_imitation_labels):
            if l in reference_label_list:
                imitations.append(i)
                imitation_labels.append(l)

        return np.array(imitations), np.array(imitation_labels)


class SiamesePartition:
    def __init__(self, references, reference_labels, all_imitations, all_imitation_labels, dataset_type):
        super().__init__()
        self.references = references
        self.reference_labels = reference_labels

        reference_label_list = [v['label'] for v in reference_labels]

        # filter out imitations of things that are not in this set at all
        self.imitations = []
        self.imitation_labels = []
        for i, l in zip(all_imitations, all_imitation_labels):
            if l in reference_label_list:
                self.imitations.append(i)
                self.imitation_labels.append(l)

        bar = Bar("Creating pairs for {0}...".format(dataset_type), max=len(self.references) * len(self.imitations))
        self.positive_pairs = []
        self.negative_pairs = []
        self.all_pairs = []
        self.all_labels = np.zeros([len(self.imitations), len(self.references)])
        self.canonical_labels = np.zeros([len(self.imitations), len(self.references)])
        n = 0
        update_bar_every = 1000
        for i, (imitation, imitation_label) in enumerate(zip(self.imitations, self.imitation_labels)):
            for j, (reference, reference_label) in enumerate(zip(self.references, self.reference_labels)):
                if reference_label['label'] == imitation_label and reference_label['is_canonical']:
                    self.positive_pairs.append([imitation, reference, True])
                    self.all_pairs.append([imitation, reference, True])
                    self.all_labels[i, j] = 1
                    if reference_label['is_canonical']:
                        self.canonical_labels[i, j] = 1
                else:
                    self.negative_pairs.append([imitation, reference, False])
                    self.all_pairs.append([imitation, reference, False])
                    self.all_labels[i, j] = 0

                n += 1
                if n % update_bar_every == 0:
                    bar.next(n=update_bar_every)
        bar.finish()
