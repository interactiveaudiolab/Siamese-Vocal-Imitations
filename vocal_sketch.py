import numpy as np
import torch.utils.data.dataset as dataset

from datafiles import SegmentedDataFiles


class FineTuned(dataset.Dataset):
    def __init__(self, data_files: SegmentedDataFiles):
        self.all_positives = data_files.positive_pairs
        self.references = data_files.references
        self.imitations = data_files.imitations
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
    def __init__(self, data_files: SegmentedDataFiles):
        self.positives = data_files.positive_pairs
        self.negatives = data_files.negative_pairs
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
    def __init__(self, data_files: SegmentedDataFiles):
        self.imitations = data_files.imitations
        self.references = data_files.references

        self.n_imitations = len(self.imitations)
        self.n_references = len(self.references)
        self.labels = data_files.labels

        self.pairs = data_files.positive_pairs + data_files.negative_pairs

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)
