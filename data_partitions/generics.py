class Partition:
    def __init__(self):
        pass


class DataSplit:
    def __init__(self, train_ratio: float, validation_ratio: float, test_ratio: float = None):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        if test_ratio:
            self.test_ratio = test_ratio
        else:
            self.test_ratio = 1 - (train_ratio + validation_ratio)