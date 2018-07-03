import logging

import numpy as np
from progress.bar import Bar

from utils import utils


class VocalSketch:
    def __init__(self, train_ratio, val_ratio, test_ratio, shuffle=True):
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Training, validation, and testing ratios must add to 1")

        logger = logging.getLogger('logger')
        logger.info("train, validation, test ratios = {0}, {1}, {2}".format(train_ratio, val_ratio, test_ratio))

        references = utils.load_npy("references.npy")
        reference_labels = utils.load_npy("reference_labels.npy")

        imitations = utils.load_npy("imitations.npy")
        imitation_labels = utils.load_npy("imitation_labels.npy")

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
        train_val_imit, train_val_imit_lab = filter_imitations(imitations, imitation_labels, train_val_ref_labels)
        n_train_imit = int(train_ratio / (train_ratio + val_ratio) * len(train_val_imit))
        train_imit = train_val_imit[:n_train_imit]
        val_imit = train_val_imit[n_train_imit:]
        train_imit_labels = train_val_imit_lab[:n_train_imit]
        val_imit_labels = train_val_imit_lab[n_train_imit:]

        self.train = VocalSketchPartition(train_val_ref, train_val_ref_labels, train_imit, train_imit_labels, "training")
        self.val = VocalSketchPartition(train_val_ref, train_val_ref_labels, val_imit, val_imit_labels, "validation")
        self.test = VocalSketchPartition(test_ref, test_ref_labels, imitations, imitation_labels, "testing")


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


def filter_imitations(all_imitations, all_imitation_labels, reference_labels):
    imitations = []
    imitation_labels = []
    for i, l in zip(all_imitations, all_imitation_labels):
        if l in reference_labels:
            imitations.append(i)
            imitation_labels.append(l)

    return imitations, imitation_labels


def zip_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
