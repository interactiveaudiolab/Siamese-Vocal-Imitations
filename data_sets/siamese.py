import math

import numpy as np
import torch.utils.data.dataset as dataset
from torch.utils.data.sampler import Sampler

from data_partitions.siamese import SiamesePartition


class SiameseDataset(dataset.Dataset):
    def __init__(self, data):
        self.pairs = []

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    def epoch_handler(self):
        pass


class FineTuned(SiameseDataset):
    def __init__(self, data: SiamesePartition):
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


class AllPositivesRandomNegatives(SiameseDataset):
    def __init__(self, data: SiamesePartition):
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
            self.pairs.append([imitation, reference, label])

        for imitation, reference, label in self.positives:
            self.pairs.append([imitation, reference, label])

        np.random.shuffle(self.pairs)


class AllPairs(SiameseDataset):
    def __init__(self, data: SiamesePartition):
        super().__init__(data)
        self.imitations = data.imitations
        self.references = data.references

        self.n_imitations = len(self.imitations)
        self.n_references = len(self.references)
        self.canonical_labels = data.canonical_labels
        self.all_labels = data.all_labels

        self.pairs = data.all_pairs


class BalancedSampler(Sampler):
    def __init__(self, data: SiameseDataset, batch_size, drop_last=False):
        super().__init__(data)
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        pos_i = 0
        neg_i = 0
        half = self.batch_size / 2
        for batch_n in range(self.__len__()):
            batch = []
            while len(batch) < half and pos_i < self.data.__len__():
                item = self.data.__getitem__(pos_i)
                if item[2]:
                    batch.append(pos_i)
                pos_i += 1

            while len(batch) < self.batch_size and neg_i < self.data.__len__():
                item = self.data.__getitem__(neg_i)
                if not item[2]:
                    batch.append(neg_i)
                neg_i += 1

            batches += batch

        return iter(batches)

    def __len__(self):
        n_batches = self.data.__len__() / self.batch_size
        if self.drop_last:
            return math.floor(n_batches)
        else:
            return math.ceil(n_batches)
