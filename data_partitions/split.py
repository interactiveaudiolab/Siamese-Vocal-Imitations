import os

import numpy as np


class PartitionSplit:
    def __init__(self, train_ratio: float, validation_ratio: float, test_ratio: float = None, train_val_n: int = None):
        super().__init__()
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        if test_ratio:
            self.test_ratio = test_ratio
        else:
            self.test_ratio = 1 - (train_ratio + validation_ratio)

        self.train_val_n = train_val_n

    def split_categories(self, categories):
        if self.train_val_n:
            train_val_n = self.train_val_n
        else:
            train_val_n = int((self.train_ratio + self.validation_ratio) * len(categories))
        return categories[:train_val_n], categories[train_val_n:]

    def split_imitations(self, imitations):
        n_imitations = len(set(t.index for t in imitations))
        train_n = int(self.train_ratio / (self.train_ratio + self.validation_ratio) * n_imitations)
        train_range = np.arange(train_n)
        train = []
        val = []
        for imitation in imitations:
            if imitation.index in train_range:
                train.append(imitation)
            else:
                val.append(imitation)

        return train, val

    @staticmethod
    def _get_imitation_index(imitation):
        return os.path.basename(imitation).split("_")[1]
