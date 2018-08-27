from data_partitions.partitions import Partitions


class Experiment:
    def __init__(self, use_cuda: bool, n_epochs: int, validate_every: int, use_dropout: bool, partitions: Partitions,
                 optimizer_name: str, lr: float, wd: float, momentum: bool):
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.validate_every = validate_every
        self.use_dropout = use_dropout
        self.partitions = partitions
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.wd = wd
        self.momentum = momentum

    def __call__(self):
        self.train()

    def train(self):
        raise NotImplementedError
