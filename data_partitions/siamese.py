import logging

import numpy as np

from data_files.generics import Datafiles
from utils.obj import DataSplit
from utils.progress_bar import Bar


class SiamesePartitions:
    def __init__(self, dataset: Datafiles, split: DataSplit):
        logger = logging.getLogger('logger')
        logger.info("train, validation, test ratios = {0}, {1}, {2}".format(split.train_ratio, split.validation_ratio, split.test_ratio))

        imitations = dataset.imitations
        imitation_labels = dataset.imitation_labels
        references = dataset.references
        reference_labels = dataset.reference_labels

        # Split categories across train/val and test
        # i.e. test has test_ratio categories
        unique_labels = list(set([v['label'] for v in reference_labels]))
        np.random.shuffle(unique_labels)
        n_train_val = int(split.train_ratio * len(unique_labels) + split.validation_ratio * len(unique_labels))
        train_val_categories = unique_labels[:n_train_val]
        test_categories = unique_labels[n_train_val:]

        train_val_ref, train_val_ref_labels = self.filter_references(references, reference_labels, train_val_categories)
        test_ref, test_ref_labels = self.filter_references(references, reference_labels, test_categories)

        train_val_imit, train_val_imit_labels = self.filter_imitations(imitations, imitation_labels, train_val_ref_labels)
        try:
            n_train_imit = int(split.train_ratio / (split.train_ratio + split.validation_ratio) * len(train_val_imit))
        except ZeroDivisionError:
            n_train_imit = 0

        train_imit = train_val_imit[:n_train_imit]
        val_imit = train_val_imit[n_train_imit:]
        train_imit_labels = train_val_imit_labels[:n_train_imit]
        val_imit_labels = train_val_imit_labels[n_train_imit:]

        self.train = SiamesePartition(train_val_ref, train_val_ref_labels, train_imit, train_imit_labels, "training")
        self.val = SiamesePartition(train_val_ref, train_val_ref_labels, val_imit, val_imit_labels, "validation")
        self.test = SiamesePartition(test_ref, test_ref_labels, imitations, imitation_labels, "testing")

    @staticmethod
    def filter_references(references, reference_labels, categories):
        ref = []
        ref_labels = []
        for r, l in zip(references, reference_labels):
            if l['label'] in categories:
                ref.append(r)
                ref_labels.append(l)
        return ref, ref_labels

    @staticmethod
    def filter_imitations(all_imitations, all_imitation_labels, reference_labels):
        imitations = []
        imitation_labels = []
        reference_label_list = [v['label'] for v in reference_labels]
        for i, l in zip(all_imitations, all_imitation_labels):
            if l in reference_label_list:
                imitations.append(i)
                imitation_labels.append(l)

        return imitations, imitation_labels


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
