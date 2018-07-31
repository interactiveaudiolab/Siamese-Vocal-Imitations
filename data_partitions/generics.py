import logging
import pickle
import sys

import numpy as np

from data_files.generics import Datafiles
from utils.obj import DataSplit
from utils.progress_bar import Bar


class Partitions:
    def __init__(self, dataset: Datafiles, split: DataSplit, n_train_val_categories=None, regenerate_splits=False):
        dataset_name = type(dataset).__name__
        pickle_name = "./partition_pickles/{0}.pickle".format(dataset_name)
        logger = logging.getLogger('logger')

        self.train = None
        self.val = None
        self.test = None

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

            if not n_train_val_categories:
                n_train_val = int(split.train_ratio * len(categories) + split.validation_ratio * len(categories))
            else:
                n_train_val = n_train_val_categories

            train_val_ref, train_val_ref_labels = self.split_references(references, reference_labels, categories[:n_train_val])
            test_ref, test_ref_labels = self.split_references(references, reference_labels, categories[n_train_val:])

            train_val_imit, train_val_imit_lab = self.filter_imitations(imitations, imitation_labels, train_val_ref_labels)

            train_imit, train_imit_lab, val_imit, val_imit_lab = self.split_imitations(categories, imitations, split, train_val_imit, train_val_imit_lab)

            self.train_args = [train_val_ref, train_val_ref_labels, train_imit, train_imit_lab, "training"]
            self.val_args = [train_val_ref, train_val_ref_labels, val_imit, val_imit_lab, "validation"]
            self.test_args = [test_ref, test_ref_labels, imitations, imitation_labels, "testing"]

            logger.debug("Saving partitions at {0}...".format(pickle_name))
            with open(pickle_name, 'wb') as f:
                pickle.dump(self.train_args, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.val_args, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.test_args, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            try:
                logger.debug("Loading partitions from {0}...".format(pickle_name))
                with open(pickle_name, 'rb') as f:
                    self.train_args = pickle.load(f)
                    self.val_args = pickle.load(f)
                    self.test_args = pickle.load(f)
            except FileNotFoundError:
                with open(pickle_name, 'w+b'):
                    logger.critical("No pickled partition at {0}".format(pickle_name))
                    sys.exit()
            except EOFError:
                logger.critical("Insufficient amount of data found in {0}".format(pickle_name))
                sys.exit()

    def generate_partitions(self, partition, no_test=False, train_only=False):
        self.train = partition(*self.train_args)
        if not train_only:
            self.val = partition(*self.val_args)
            if not no_test:
                self.test = partition(*self.test_args)

    @staticmethod
    def split_imitations(categories, imitations, split, train_val_imit, train_val_imit_labels):
        imitation_shape = list(imitations.shape)
        imitation_shape[0] = 0
        train_imit = np.empty(imitation_shape)
        val_imit = np.empty(imitation_shape)
        train_imit_labels = np.empty(0)
        val_imit_labels = np.empty(0)
        bar = Bar("Splitting imitations by category...", max=len(categories))
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
            bar.next()
        bar.finish()
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


class Partition:
    def __init__(self):
        pass
