import math

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from data_sets.siamese import BalancedSampler, SiameseDataset
from models.siamese import Siamese
from utils.progress_bar import Bar


def train_siamese_network(model: Siamese, data: SiameseDataset, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # because the model is passed by reference and this is a generator, ensure that we're back in training mode
        model = model.train()

        # notify the dataset that an epoch has passed
        data.epoch_handler()

        batch_sampler = BatchSampler(BalancedSampler(data, batch_size), batch_size=batch_size, drop_last=False)
        train_data = DataLoader(data, batch_sampler=batch_sampler, num_workers=1)

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
            loss.backward()
            optimizer.step()
            batch_losses[i] = loss.item()

            bar.next()
        bar.finish()

        yield model, batch_losses
