import logging

import numpy as np
from progress.bar import Bar

from utils import utils


class VocalSketch:
    def __init__(self, train_ratio, val_ratio, test_ratio):
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Training, validation, and testing ratios must add to 1")

        logger = logging.getLogger('logger')
        logger.info("train, validation, test ratios = {0}, {1}, {2}".format(train_ratio, val_ratio, test_ratio))

        references = utils.load_npy("references.npy")
        reference_labels = utils.load_npy("reference_labels.npy")
        references, reference_labels = zip_shuffle(references, reference_labels)

        imitations = utils.load_npy("imitations.npy")
        imitation_labels = utils.load_npy("imitation_labels.npy")
        imitations, imitation_labels = zip_shuffle(imitations, imitation_labels)

        n_train = int(train_ratio * len(references))
        n_val = int(val_ratio * len(references))
        n_test = int(test_ratio * len(references))

        train_ref = references[:n_train]
        val_ref = references[n_train:n_train + n_val]
        test_ref = references[n_train + n_val:]

        train_ref_labels = reference_labels[:n_train]
        val_ref_labels = reference_labels[n_train:n_train + n_val]
        test_ref_labels = reference_labels[n_train + n_val:]

        self.train = VocalSketchPartition(train_ref, train_ref_labels, imitations, imitation_labels, "training")
        self.val = VocalSketchPartition(val_ref, val_ref_labels, imitations, imitation_labels, "validation")
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
        self.labels = np.zeros([len(self.imitations), len(self.references)])
        for i, (imitation, imitation_label) in enumerate(zip(self.imitations, self.imitation_labels)):
            for j, (reference, reference_label) in enumerate(zip(self.references, self.reference_labels)):
                if reference_label == imitation_label:
                    self.positive_pairs.append([imitation, reference, True])
                    self.labels[i, j] = 1
                else:
                    self.negative_pairs.append([imitation, reference, False])
                    self.labels[i, j] = 0

                bar.next()
        bar.finish()


def zip_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
