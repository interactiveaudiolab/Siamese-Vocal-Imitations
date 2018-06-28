import numpy as np
import torch.utils.data.dataset as dataset

from datafiles.vocal_sketch_files import VocalSketchPartition


class FineTuned(dataset.Dataset):
    def __init__(self, data: VocalSketchPartition):
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

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


class AllPositivesRandomNegatives(dataset.Dataset):
    def __init__(self, data: VocalSketchPartition):
        self.positives = data.positive_pairs
        self.negatives = data.negative_pairs
        self.pairs = []
        self.reselect_negatives()

    def reselect_negatives(self):
        # clear out selected negatives
        self.pairs = []
        indices = np.random.choice(np.arange(len(self.negatives)), len(self.positives))
        for i in indices:
            imitation, reference, label = self.negatives[i]
            self.pairs.append([imitation, reference, label])

        for imitation, reference, label in self.positives:
            self.pairs.append([imitation, reference, label])

        np.random.shuffle(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


class AllPairs(dataset.Dataset):
    def __init__(self, data: VocalSketchPartition):
        self.imitations = data.imitations
        self.references = data.references

        self.n_imitations = len(self.imitations)
        self.n_references = len(self.references)
        self.labels = data.labels

        self.pairs = data.positive_pairs + data.negative_pairs

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)
