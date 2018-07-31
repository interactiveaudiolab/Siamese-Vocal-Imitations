import numpy as np

from data_partitions.pair import PairPartition
from data_sets.generics import PairedDataset


class AllPositivesRandomNegatives(PairedDataset):
    def __init__(self, data: PairPartition):
        super().__init__(data)
        self.positives = data.positive_pairs
        self.negatives = data.negative_pairs
        self.pairs = []
        self.reselect_negatives()

    def epoch_handler(self):
        self.reselect_negatives()

    def reselect_negatives(self):
        # clear out selected negatives
        self.pairs = []
        indices = np.random.choice(np.arange(len(self.negatives)), len(self.positives))
        for i in indices:
            imitation, reference, label = self.negatives[i]
            b = self.negatives[i]
            self.pairs.append([imitation, reference, label])

        for imitation, reference, label in self.positives:
            self.pairs.append([imitation, reference, label])

        np.random.shuffle(self.pairs)


class AllPairs(PairedDataset):
    def __init__(self, data: PairPartition):
        super().__init__(data)
        self.imitations = data.imitations
        self.references = data.references

        self.n_imitations = len(self.imitations)
        self.n_references = len(self.references)
        self.canonical_labels = data.canonical_labels
        self.all_labels = data.all_labels

        self.pairs = data.all_pairs
