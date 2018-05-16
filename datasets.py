import numpy as np
import torch.utils.data.dataset as dataset


class VocalImitations(dataset.Dataset):
    def __init__(self, is_train=True, max_size=None):
        if is_train:
            left = np.load('./data/train_pairs_data_left.npy')[:max_size]
            right = np.load('./data/train_pairs_data_right.npy')[:max_size]
            labels = np.load('./data/train_pairs_labels.npy')[:max_size]
        else:
            left = np.load('./data/val_pairs_data_left.npy')[:max_size]
            right = np.load('./data/val_pairs_data_right.npy')[:max_size]
            labels = np.load('./data/val_pairs_labels.npy')[:max_size]

        self.data = list(zip(left, right, labels))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
