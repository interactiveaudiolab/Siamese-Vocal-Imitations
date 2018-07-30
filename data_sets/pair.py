import numpy as np

from data_partitions.siamese import PairPartition
from data_sets.generics import PairedDataset


class FineTuned(PairedDataset):
    def __init__(self, data: PairPartition):
        super().__init__(data)
        self.all_positives = data.positive_pairs
        self.references = data.references
        self.imitations = data.imitations
        self.pairs = []

    def add_negatives(self, reference_indexes):
        for i, r in enumerate(reference_indexes):
            self.add_negative(self.imitations[i], self.references[r])

    def add_negative(self, i, r):
        self.pairs.append([i, r, False])

    def reset(self):
        self.pairs = []
        for i, r, l in self.all_positives:
            self.pairs.append([i, r, l])


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
