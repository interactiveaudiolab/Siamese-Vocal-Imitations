import math

from torch.utils.data.sampler import Sampler

from data_subsets import PairedDataSubset, TripletDataSubset


class BalancedPairSampler(Sampler):
    def __init__(self, data: PairedDataSubset, batch_size, drop_last=False):
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


class BalancedTripletSampler(Sampler):
    def __init__(self, data: TripletDataSubset, batch_size, drop_last=False):
        super().__init__(data)
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        pass

    def __len__(self):
        pass
