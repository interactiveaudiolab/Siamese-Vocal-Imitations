import numpy as np

from data_subsets import PairedDataSubset
from data_partitions import PairPartition


class Balanced(PairedDataSubset):
    def __init__(self, data: PairPartition):
        """
        Create a dataset that has an equal number of positive and negative examples and an equal number of negative fine grain examples and negative coarse
        grain examples. Reselects the negative examples on each epoch.

        :param data:
        """
        super().__init__(data)
        self.positive = data.positive
        self.negative_coarse = data.negative_coarse
        self.negative_fine = data.negative_fine
        self.reselect_negatives()

    def epoch_handler(self):
        self.reselect_negatives()

    def reselect_negatives(self):
        # clear out selected negatives
        self.pairs = []

        negative_size = int(len(self.positive) / 2)
        pairs = np.concatenate(self.positive,
                               np.random.choice(self.negative_fine, negative_size),
                               np.random.choice(self.negative_coarse, negative_size))
        for i, r, l in pairs:
            self.pairs.append([i.load(), r.load(), l])

        np.random.shuffle(self.pairs)


class AllPairs(PairedDataSubset):
    def __init__(self, data: PairPartition):
        super().__init__(data)
        self.imitations = data.imitations
        self.references = data.references

        self.n_imitations = len(self.imitations)
        self.n_references = len(self.references)
        self.labels = data.labels
        # self.canonical_locations = data.canonical_locations

        self.pairs = data.all_pairs
