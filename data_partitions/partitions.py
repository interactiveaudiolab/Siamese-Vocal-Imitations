from typing import Type

import numpy as np

from data_files import Datafiles
from data_partitions import PartitionArguments, SaveablePartitionState, Partition, PartitionSplit
from utils.progress_bar import Bar


class Partitions:
    def __init__(self, dataset: Datafiles, split: PartitionSplit, n_train_val_categories=None, regenerate=False):
        dataset_name = dataset.name
        pickle_name = "./partition_pickles/{0}.pickle".format(dataset_name)

        self.train_args = None
        self.val_args = None
        self.test_args = None

        self.train = None
        self.val = None
        self.test = None

        self.dataset = dataset
        self.state = SaveablePartitionState()

        if split and n_train_val_categories:
            raise RuntimeWarning("Both n_train_val_categories and a data split were provided. The data split is overriden.")

        if regenerate:
            shuffled_imitation_indices = np.random.permutation(len(dataset.imitations))

            categories = np.array(list(set([v['label'] for v in dataset.reference_labels])))
            shuffled_category_indices = np.random.permutation(len(categories))

            self.state.shuffled_category_indices = shuffled_category_indices
            self.state.shuffled_imitation_indices = shuffled_imitation_indices
            self.state.split = split
            self.state.n_train_val_categories = n_train_val_categories
            self.state.categories = categories

            self.state.save(pickle_name)
            self._construct_partition_arguments()

        else:
            self.state.load(pickle_name)
            self._construct_partition_arguments()

    def generate_partitions(self, partition: Type[Partition], no_test=False, train_only=False):
        """
        Lazily generate partitions of a given type.

        :param partition:
        :param no_test:
        :param train_only:
        """
        self.train = partition(self.train_args.references, self.train_args.reference_labels, self.train_args.imitations, self.train_args.imitation_labels)
        if not train_only:
            self.val = partition(self.val_args.references, self.val_args.reference_labels, self.val_args.imitations, self.val_args.imitation_labels)
            if not no_test:
                self.test = partition(self.test_args.references, self.test_args.reference_labels, self.test_args.imitations, self.test_args.imitation_labels)

    def save(self, location):
        self.state.save(location)

    def _construct_partition_arguments(self):
        imitations = self.dataset.imitations
        imitation_labels = self.dataset.imitation_labels
        references = self.dataset.references
        reference_labels = self.dataset.reference_labels

        # Split categories across train/val and test
        # i.e. train/val share a set of categories and test uses the other ones
        categories = np.array(list(set([v['label'] for v in reference_labels])))
        categories = categories[self.state.shuffled_category_indices]
        if not self.state.n_train_val_categories:
            n_train_val = int(self.state.split.train_ratio * len(categories) + self.state.split.validation_ratio * len(categories))
        else:
            n_train_val = self.state.n_train_val_categories

        train_val_ref, train_val_ref_labels = self._split_references(references, reference_labels, categories[:n_train_val])
        test_ref, test_ref_labels = self._split_references(references, reference_labels, categories[n_train_val:])

        imitations = imitations[self.state.shuffled_imitation_indices]
        imitation_labels = imitation_labels[self.state.shuffled_imitation_indices]
        train_val_imit, train_val_imit_lab = self._filter_imitations(imitations, imitation_labels, train_val_ref_labels)
        train_imit, train_imit_lab, val_imit, val_imit_lab = self._split_imitations(categories,
                                                                                    imitations,
                                                                                    self.state.split,
                                                                                    train_val_imit,
                                                                                    train_val_imit_lab)

        self.train_args = PartitionArguments(train_val_ref, train_val_ref_labels, train_imit, train_imit_lab, "training")
        self.val_args = PartitionArguments(train_val_ref, train_val_ref_labels, val_imit, val_imit_lab, "validation")
        self.test_args = PartitionArguments(test_ref, test_ref_labels, imitations, imitation_labels, "testing")

    @staticmethod
    def _split_imitations(categories, imitations, split, train_val_imit, train_val_imit_labels):
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
    def _split_references(references, reference_labels, categories):
        ref = []
        ref_labels = []
        for r, l in zip(references, reference_labels):
            if l['label'] in categories:
                ref.append(r)
                ref_labels.append(l)
        return np.array(ref), np.array(ref_labels)

    @staticmethod
    def _filter_imitations(all_imitations, all_imitation_labels, reference_labels):
        imitations = []
        imitation_labels = []
        reference_label_list = [v['label'] for v in reference_labels]
        for i, l in zip(all_imitations, all_imitation_labels):
            if l in reference_label_list:
                imitations.append(i)
                imitation_labels.append(l)

        return np.array(imitations), np.array(imitation_labels)
