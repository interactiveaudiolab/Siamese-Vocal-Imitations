import numpy as np
from progress.bar import Bar
from torch.utils.data import DataLoader

from datasets.urban_sound_8k import UrbanSound10FCV
from datasets.vocal_sketch_data import AllPositivesRandomNegatives


def train_siamese_network(model, data, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # if we're using all positives and random negatives, choose new negatives on each epoch
        if isinstance(data, AllPositivesRandomNegatives):
            data.reselect_negatives()

        train_data = DataLoader(data, batch_size=batch_size, num_workers=1)
        bar = Bar("Training siamese, epoch {0}".format(epoch), max=len(train_data))
        batch_losses = np.zeros(len(train_data))
        for i, (left, right, labels) in enumerate(train_data):
            # clear out the gradients
            optimizer.zero_grad()

            # TODO: make them floats at the source
            labels = labels.float()

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


def train_right_tower(model, data: UrbanSound10FCV, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        train_data = DataLoader(data, batch_size=batch_size, num_workers=1)
        bar = Bar("Training right tower, epoch {0}".format(epoch), max=len(train_data))
        batch_losses = np.zeros(len(train_data))
        for i, (audio, labels) in enumerate(train_data):
            # clear out the gradients
            optimizer.zero_grad()

            # reshape tensors and push to GPU if necessary
            audio = audio.unsqueeze(1)
            if use_cuda:
                audio = audio.cuda()
                labels = labels.cuda()

            # pass a batch through the network
            outputs = model(audio)

            # calculate loss and optimize weights
            labels = labels.long()
            loss = objective(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_losses[i] = loss.item()

            bar.next()
        bar.finish()

        yield model, batch_losses
