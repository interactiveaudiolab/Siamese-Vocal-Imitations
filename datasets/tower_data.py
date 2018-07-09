from torch.utils.data import dataset


class TowerData(dataset.Dataset):
    def __init__(self):
        self.validating = False
        self.validation = []
        self.training = []

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
            return self.validation[index]
        else:
            return self.training[index]

    def __len__(self):
        if self.validating:
            return len(self.validation)
        else:
            return len(self.training)
