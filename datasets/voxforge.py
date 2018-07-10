from datafiles.voxforge import Voxforge
from datasets.tower_data import TowerData
from utils.utils import zip_shuffle


class All(TowerData):
    def __init__(self, data: Voxforge, shuffle=True):
        super().__init__()
        self.labels = data.labels
        self.unique_labels = data.unique_labels
        self.spectrograms = data.spectrograms

        if shuffle:
            self.labels, self.spectrograms = zip_shuffle(self.labels, self.spectrograms)

        self.by_label = {u: [] for u in self.unique_labels}

        n_train = int(.7 * data.per_language)
        n_validate = int(.3 * data.per_language)

        for unique_label in self.unique_labels:
            for spectrogram, label in zip(self.spectrograms, self.labels):
                if label == unique_label:
                    self.by_label[label].append([spectrogram, label])

        for unique_label in self.unique_labels:
            language = self.by_label[unique_label]
            self.training += language[:n_train]
            self.validation += language[n_train:n_train + n_validate]

    def __getitem__(self, index):
        if self.validating:
            return self.validation[index]
        else:
            return self.training[index]

    def __len__(self):
        if self.validating:
            return len(self.validation)
        else:
            return len(self.training)
