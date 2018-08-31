import numpy as np

from data_subsets import TripletDataSubset
from data_partitions import TripletPartition


class Balanced(TripletDataSubset):
    def __init__(self, data: TripletPartition):
        super().__init__(data)
        self.positive_coarse = data.positive_coarse
        self.positive_fine = data.positive_fine
        self.negative_coarse = data.negative_coarse
        self.negative_fine = data.negative_fine

        self.reselect_coarse()

    def epoch_handler(self):
        self.reselect_coarse()

    def reselect_coarse(self):
        # clear out selected negatives
        self.triplets = []

        triplets = np.concatenate((self.positive_fine,
                                   self.negative_fine,
                                   # select random subset of positive coarse grain examples equal in size to positive fine grain examples
                                   (np.random.choice(self.positive_coarse, len(self.positive_fine))),
                                   # select random subset of negative coarse grain examples equal in size to negative fine grain examples
                                   (np.random.choice(self.negative_coarse, len(self.negative_fine)))))

        # load the spectrograms into memory
        for i, n, f, l in triplets:
            self.triplets.append([i.load(), n.load(), f.load(), l])

        np.random.shuffle(self.triplets)
