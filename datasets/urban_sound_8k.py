from torch.utils.data import dataset

from datafiles.urban_sound_8k import UrbanSound8K


class UrbanSound10FCV(dataset.Dataset):
    def __init__(self, data: UrbanSound8K):
        self.folds = data.folds
        self.n_folds = len(data.folds)
        self.fold_labels = data.fold_labels
        self.validating = False
        self.current_fold = None
        self.train_data = None
        self.validation_data = None

    def set_fold(self, fold):
        self.current_fold = fold
        self.train_data = []
        self.validation_data = []
        for d, l in zip(self.folds[self.current_fold], self.fold_labels[self.current_fold]):
            self.validation_data.append([d, l])

        for i in [j for j in range(len(self.folds)) if j != self.current_fold]:
            for d, l in zip(self.folds[i], self.fold_labels[i]):
                self.train_data.append([d, l])

    def validation_mode(self):
        if self.validating:
            raise RuntimeWarning("Redundant switch to validation mode")
        self.validating = True

    def training_mode(self):
        if not self.validating:
            raise RuntimeWarning("Redundant switch to training mode")
        self.validating = False

    def __getitem__(self, index):
        if self.validating:
            return self.validation_data[index]
        else:
            return self.train_data[index]

    def __len__(self):
        if self.validating:
            return len(self.validation_data)
        else:
            return len(self.train_data)
