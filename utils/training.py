import math

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from data_subsets import PairedDataSubset, TripletDataSubset
from data_subsets.samplers import BalancedPairSampler
from models.siamese import Siamese
from models.triplet import Triplet
from utils.progress_bar import Bar


def train_siamese_network(model: Siamese, data: PairedDataSubset, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # because the model is passed by reference and this is a generator, ensure that we're back in training mode
        model = model.train()

        # notify the dataset that an epoch has passed
        data.epoch_handler()

        batch_sampler = BatchSampler(BalancedPairSampler(data, batch_size), batch_size=batch_size, drop_last=False)
        train_data = DataLoader(data, batch_sampler=batch_sampler, num_workers=4)

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


def train_triplet_network(model: Triplet, data: TripletDataSubset, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # because the model is passed by reference and this is a generator, ensure that we're back in training mode
        model = model.train()
        # notify the dataset that an epoch has passed
        data.epoch_handler()

        # batch_sampler = BatchSampler(BalancedTripletSampler(data, batch_size), batch_size=batch_size, drop_last=False)
        train_data = DataLoader(data, batch_size=batch_size, num_workers=4)

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
