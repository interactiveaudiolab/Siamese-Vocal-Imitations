from typing import List

import numpy as np

from data_partitions import Imitation, Reference
from utils.progress_bar import Bar


class PairPartition:
    def __init__(self, imitations: List[Imitation], references: List[Reference]):
        self.imitations = imitations
        self.references = references

        self.positive = []
        self.negative_coarse = []
        self.negative_fine = []
        self.all_pairs = []
        self.labels = np.zeros(shape=[len(imitations), len(references)])

        bar = Bar("Creating pairs...", max=len(imitations) * len(references))
        for i, imitation in enumerate(self.imitations):
            for j, reference in enumerate(self.references):
                label = self._classify_pair(imitation, reference)
                self.all_pairs.append([imitation, reference, label])
                self.labels[i, j] = label
                bar.next()

        bar.finish()

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
