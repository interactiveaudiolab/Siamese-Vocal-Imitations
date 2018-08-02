import logging
import math
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from data_sets.generics import PairedDataset, TripletDataset
from data_sets.samplers import BalancedPairSampler
from models.siamese import Siamese
from models.triplet import Triplet
from utils.graphing import mean_rank_per_epoch, mrr_per_epoch, loss_per_epoch
from utils.progress_bar import Bar
from utils.utils import get_trial_number


def train_siamese_network(model: Siamese, data: PairedDataset, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # because the model is passed by reference and this is a generator, ensure that we're back in training mode
        model = model.train()

        # notify the dataset that an epoch has passed
        data.epoch_handler()

        batch_sampler = BatchSampler(BalancedPairSampler(data, batch_size), batch_size=batch_size, drop_last=False)
        train_data = DataLoader(data, batch_sampler=batch_sampler, num_workers=2)

        train_data_len = math.ceil(train_data.dataset.__len__() / batch_size)
        batch_losses = np.zeros(train_data_len)
        bar = Bar("Training siamese, epoch {0}".format(epoch), max=train_data_len)
        for i, (left, right, labels) in enumerate(train_data):
            # clear out the gradients
            optimizer.zero_grad()

            labels = labels.float()
            left = left.float()
            right = right.float()

            # reshape tensors and push to GPU if necessary
            left = left.unsqueeze(1)
            right = right.unsqueeze(1)
            if use_cuda:
                left = left.cuda()
                right = right.cuda()
                labels = labels.cuda()

            # pass a batch through the network
            outputs = model(left, right)

            # calculate loss and optimize weights
            loss = objective(outputs, labels)
            batch_losses[i] = loss.item()
            loss.backward()
            optimizer.step()

            bar.next()
        bar.finish()

        yield model, batch_losses


def train_triplet_network(model: Triplet, data: TripletDataset, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # because the model is passed by reference and this is a generator, ensure that we're back in training mode
        model = model.train()
        # notify the dataset that an epoch has passed
        data.epoch_handler()

        # batch_sampler = BatchSampler(BalancedTripletSampler(data, batch_size), batch_size=batch_size, drop_last=False)
        train_data = DataLoader(data, batch_size=batch_size, num_workers=2)

        train_data_len = math.ceil(train_data.dataset.__len__() / batch_size)
        batch_losses = np.zeros(train_data_len)
        bar = Bar("Training siamese, epoch {0}".format(epoch), max=train_data_len)
        for i, triplet in enumerate(train_data):
            # clear out the gradients
            optimizer.zero_grad()

            triplet = [tensor.float() for tensor in triplet]

            # reshape tensors and push to GPU if necessary
            triplet = [tensor.unsqueeze(1) for tensor in triplet[:3]] + [triplet[3]]
            if use_cuda:
                triplet = [tensor.cuda() for tensor in triplet]

            # pass a batch through the network
            outputs = model(*triplet[:3])

            # calculate loss and optimize weights
            loss = objective(outputs, triplet[3])
            batch_losses[i] = loss.item()
            loss.backward()
            optimizer.step()

            bar.next()
        bar.finish()

        yield model, batch_losses


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