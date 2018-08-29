import numpy as np

from data_partitions.pair_partition import PairPartition
from data_subsets import PairedDataSubset


class Balanced(PairedDataSubset):
    def __init__(self, data: PairPartition):
        """
        Create a dataset that has an equal number of positive and negative examples and an equal number of negative fine grain examples and negative coarse
        grain examples. Reselects the negative examples on each epoch.

        :param data:
        """
        super().__init__(data)
        self.positives = data.positive
        self.negative_fine = data.negative_fine
        self.negative_coarse = data.negative_coarse

        self.pairs = []
        self.reselect_negatives()

    def epoch_handler(self):
        self.reselect_negatives()

    def reselect_negatives(self):
        # clear out selected negatives
        self.pairs = []

        negative_size = int(len(self.positives) / 2)
        fine_indices = np.random.choice(np.arange(len(self.negative_fine)), negative_size)
        for i in fine_indices:
            imitation, reference, label = self.negative_fine[i]
            self.pairs.append([imitation, reference, label])

        coarse_indices = np.random.choice(np.arange(len(self.negative_coarse)), negative_size)
        for i in coarse_indices:
            imitation, reference, label = self.negative_coarse[i]
            self.pairs.append([imitation, reference, label])

        for imitation, reference, label in self.positives:
            self.pairs.append([imitation, reference, label])

        np.random.shuffle(self.pairs)


class AllPairs(PairedDataSubset):
    def __init__(self, data: PairPartition):
        super().__init__(data)
        self.imitations = data.imitations
        self.references = data.references

        self.n_imitations = len(self.imitations)
        self.n_references = len(self.references)
        self.labels = data.labels
        self.canonical_locations = data.canonical_locations

        self.pairs = data.all_pairs
