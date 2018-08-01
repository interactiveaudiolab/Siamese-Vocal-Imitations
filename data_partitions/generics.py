import logging
import os
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

        self.train_args = None
        self.val_args = None
        self.test_args = None

        self.train = None
        self.val = None
        self.test = None

        self.n_train_val_categories = n_train_val_categories
        self.split = split

        self.dataset = dataset

        if regenerate_splits:
            self.shuffled_imitation_indices = np.random.permutation(len(dataset.imitations))

            categories = np.array(list(set([v['label'] for v in dataset.reference_labels])))
            self.shuffled_category_indices = np.random.permutation(len(categories))

            self.save(pickle_name)
            self.construct_partition_arguments()

        else:
            self.shuffled_category_indices, self.shuffled_imitation_indices = self.load(pickle_name)
            self.construct_partition_arguments()

    @staticmethod
    def load(location):
        logger = logging.getLogger('logger')

        try:
            logger.debug("Loading partitions from {0}...".format(location))
            with open(location, 'rb') as f:
                shuffled_imitation_indices = pickle.load(f)
                shuffled_category_indices = pickle.load(f)

        except FileNotFoundError:
            with open(location, 'w+b'):
                logger.critical("No pickled partition at {0}".format(location))
                sys.exit()
        except EOFError:
            logger.critical("Insufficient amount of data found in {0}".format(location))
            sys.exit()
        return shuffled_category_indices, shuffled_imitation_indices

    def save(self, location):
        logger = logging.getLogger('logger')

        logger.debug("Saving partitions at {0}...".format(location))
        with open(location, 'wb') as f:
            pickle.dump(self.shuffled_imitation_indices, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.shuffled_category_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

        categories = np.array(list(set([v['label'] for v in self.dataset.reference_labels])))

        if not self.n_train_val_categories:
            n_train_val = int(self.split.train_ratio * len(categories) + self.split.validation_ratio * len(categories))
        else:
            n_train_val = self.n_train_val_categories

        categories = categories[self.shuffled_category_indices][:n_train_val]

        category_list_path = os.path.join(os.path.dirname(location), 'categories.txt')
        with open(category_list_path, 'w+') as f:
            f.write("\n".join(categories))

    def construct_partition_arguments(self):
        imitations = self.dataset.imitations
        imitation_labels = self.dataset.imitation_labels
        references = self.dataset.references
        reference_labels = self.dataset.reference_labels

        # Split categories across train/val and test
        # i.e. train/val share a set of categories and test uses the other ones
        categories = np.array(list(set([v['label'] for v in reference_labels])))
        categories = categories[self.shuffled_category_indices]
        if not self.n_train_val_categories:
            n_train_val = int(self.split.train_ratio * len(categories) + self.split.validation_ratio * len(categories))
        else:
            n_train_val = self.n_train_val_categories

        train_val_ref, train_val_ref_labels = self.split_references(references, reference_labels, categories[:n_train_val])
        test_ref, test_ref_labels = self.split_references(references, reference_labels, categories[n_train_val:])

        imitations = imitations[self.shuffled_imitation_indices]
        imitation_labels = imitation_labels[self.shuffled_imitation_indices]
        train_val_imit, train_val_imit_lab = self.filter_imitations(imitations, imitation_labels, train_val_ref_labels)
        train_imit, train_imit_lab, val_imit, val_imit_lab = self.split_imitations(categories, imitations, self.split, train_val_imit, train_val_imit_lab)

        self.train_args = [train_val_ref, train_val_ref_labels, train_imit, train_imit_lab, "training"]
        self.val_args = [train_val_ref, train_val_ref_labels, val_imit, val_imit_lab, "validation"]
        self.test_args = [test_ref, test_ref_labels, imitations, imitation_labels, "testing"]

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
