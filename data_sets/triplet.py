import numpy as np

from data_partitions.triplet import TripletPartition
from data_sets.generics import TripletDataset


class Balanced(TripletDataset):
    def __init__(self, data: TripletPartition):
        super().__init__(data)
        self.positive_fine = data.positive_fine
        self.positive_coarse = data.positive_coarse
        self.negative_fine = data.negative_fine
        self.negative_coarse = data.negative_coarse

        self.reselect_coarse()

    def epoch_handler(self):
        self.reselect_coarse()

    def reselect_coarse(self):
        # clear out selected negatives
        self.triplets = []

        # select random subset of positive coarse grain examples equal in size to positive fine grain examples
        positive_indices = np.random.choice(np.arange(len(self.positive_coarse)), len(self.positive_fine))
        [self.triplets.append(self.positive_coarse[i]) for i in positive_indices]

        # select random subset of negative coarse grain examples equal in size to negative fine grain examples
        negative_indices = np.random.choice(np.arange(len(self.negative_coarse)), len(self.negative_fine))
        [self.triplets.append(self.negative_coarse[i]) for i in negative_indices]

        # add in all the fine grain examples
        [self.triplets.append(e) for e in self.positive_fine]
        [self.triplets.append(e) for e in self.negative_fine]

        np.random.shuffle(self.triplets)
