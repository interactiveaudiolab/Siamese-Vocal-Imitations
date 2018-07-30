from torch.utils.data import dataset


class Dataset(dataset.Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def epoch_handler(self):
        pass


class PairedDataset(Dataset):
    def __init__(self, data):
        self.pairs = []

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    def epoch_handler(self):
        pass


class TripletDataset(Dataset):
    def __init__(self, data):
        self.triplets = []

    def __getitem__(self, index):
        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)

    def epoch_handler(self):
        pass
