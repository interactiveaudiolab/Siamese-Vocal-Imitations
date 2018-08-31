from typing import List

from data_partitions import Imitation, Reference
from utils.progress_bar import Bar


class TripletPartition:
    def __init__(self, imitations: List[Imitation], references: List[Reference]):
        self.imitations = imitations
        self.references = references

        self.positive_fine = []
        self.positive_coarse = []
        self.negative_fine = []
        self.negative_coarse = []

        bar = Bar("Creating triplets...", max=len(imitations) * len(references) * len(references))
        for i, imitation in enumerate(self.imitations):
            for j, near in enumerate(self.references):
                for k, far in enumerate(self.references):
                    self._classify_triplet(imitation, far, near)
                    bar.next()
        bar.finish()

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