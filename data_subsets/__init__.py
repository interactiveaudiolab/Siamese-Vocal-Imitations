from torch.utils.data import dataset


class DataSubset(dataset.Dataset):
    def __init__(self, name):
        self.name = name

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def epoch_handler(self):
        pass


class PairedDataSubset(DataSubset):
    def __init__(self, name):
        super().__init__(name)
        self.pairs = []

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    def epoch_handler(self):
        pass


class TripletDataSubset(DataSubset):
    def __init__(self, name):
        super().__init__(name)
        self.triplets = []

    def __getitem__(self, index):
        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)

    def epoch_handler(self):
        pass
