import logging
import os
import pickle

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from utils.graphing import mean_rank_per_epoch, loss_per_epoch, mrr_per_epoch
from utils.utils import get_trial_number


class DataSplit:
    def __init__(self, train_ratio: float, validation_ratio: float, test_ratio: float = None):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        if test_ratio:
            self.test_ratio = test_ratio
        else:
            self.test_ratio = 1 - (train_ratio + validation_ratio)


class TrainingProgress:
    def __init__(self):
        self.train_mrr = []
        self.train_rank = []
        self.train_loss = []
        self.val_mrr = []
        self.val_rank = []
        self.val_loss = []
        self.logger = logging.getLogger('logger')

    def add_mrr(self, train, val):
        self.train_mrr.append(train)
        self.val_mrr.append(val)
        self.logger.info("MRR at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(len(self.train_mrr), train, val))

    def add_rank(self, train, val):
        if train:
            self.train_rank.append(train)
        if val:
            self.val_rank.append(val)
        self.logger.info("Mean rank at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(len(self.train_rank), train, val))

    def add_loss(self, train, val):
        if train:
            self.train_loss.append(train)
        if val:
            self.val_loss.append(val)
        self.logger.info("Loss at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(len(self.train_loss), train, val))

    def pearson(self, log=False):
        train = pearsonr(self.train_loss, self.train_rank)[0]
        val = pearsonr(self.val_loss, self.val_rank)[0]

        if log:
            self.logger.info("Correlations between loss and rank:\n\ttrn = {0}\n\tval = {1}".format(train, val))

        return train, val

    def graph(self, trial_name, search_length):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        fig.set_size_inches(16, 10)
        mean_rank_per_epoch(self.train_rank, self.val_rank, search_length, ax1)
        mrr_per_epoch(self.train_mrr, self.val_mrr, ax2, n_categories=search_length)
        loss_per_epoch(self.train_loss, self.val_loss, ax3, log=True)
        loss_per_epoch(self.train_loss, self.val_loss, ax4, log=False)

        fig.suptitle("{0}, Trial #{1}".format(trial_name, get_trial_number()))
        fig.savefig(self.filename(trial_name), dpi=200)
        plt.close()

    @staticmethod
    def filename(title):
        file = title.replace(' ', '_').replace('.', '').replace(',', '')
        file += '.png'
        file = file.lower()
        return os.path.join('./output', str(get_trial_number()), file)

    def save(self, path):
        with open(path, 'w+b') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
