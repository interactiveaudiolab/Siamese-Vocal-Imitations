from scipy.stats import pearsonr


class DataSplit:
    def __init__(self, train_ratio: float, validation_ratio: float, test_ratio: float = None):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        if test_ratio:
            self.test_ratio = test_ratio
        else:
            self.test_ratio = 1 - (train_ratio + validation_ratio)


class TrainingResult:
    def __init__(self, train_mrr, train_rank, train_loss, val_mrr, val_rank, val_loss):
        self.train_mrr = train_mrr
        self.train_rank = train_rank
        self.train_loss = train_loss
        self.val_mrr = val_mrr
        self.val_rank = val_rank
        self.val_loss = val_loss

    def pearson(self):
        train = pearsonr(self.train_loss, self.train_mrr)
        val = pearsonr(self.val_loss, self.val_loss)
        return train, val
