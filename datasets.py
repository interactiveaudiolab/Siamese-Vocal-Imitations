import numpy as np
import torch
import torch.utils.data.dataset as dataset
from torch.utils.data.sampler import Sampler


class VocalImitations(dataset.Dataset):
    def __init__(self, left, right, labels):
        self.data = list(zip(left, right, labels))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_data(is_train=True, max_size=None):
    if is_train:
        left = np.load('./data/train_pairs_data_left.npy')[:max_size]
        right = np.load('./data/train_pairs_data_right.npy')[:max_size]
        labels = np.load('./data/train_pairs_labels.npy')[:max_size]
    else:
        left = np.load('./data/val_pairs_data_left.npy')[:max_size]
        right = np.load('./data/val_pairs_data_right.npy')[:max_size]
        labels = np.load('./data/val_pairs_labels.npy')[:max_size]

    return left, right, labels


class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, subset_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.subset_size = subset_size

    def __iter__(self):
        return iter(torch.randperm(self.subset_size).tolist())

    def __len__(self):
        return self.subset_size
