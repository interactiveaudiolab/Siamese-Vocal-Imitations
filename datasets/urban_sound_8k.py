from datafiles.urban_sound_8k import UrbanSound8K
from datasets.generics import TowerData


class UrbanSound10FCV(TowerData):
    def __init__(self, data: UrbanSound8K):
        super().__init__()
        self.folds = data.folds
        self.n_folds = len(data.folds)
        self.fold_labels = data.fold_labels
        self.current_fold = None

    def set_fold(self, fold):
        self.current_fold = fold
        self.training = []
        self.validation = []
        for d, l in zip(self.folds[self.current_fold], self.fold_labels[self.current_fold]):
            self.validation.append([d, l])

        for i in [j for j in range(len(self.folds)) if j != self.current_fold]:
            for d, l in zip(self.folds[i], self.fold_labels[i]):
                self.training.append([d, l])

