import numpy as np
import torch
from progress.bar import Bar
from torch.utils.data import dataloader, DataLoader

from datasets.tower_data import TowerData
from datasets.vocal_sketch import AllPairs
from models.siamese import Siamese
from models.transfer_learning import Tower
from utils import utils


def mean_reciprocal_ranks(model: Siamese, pairs: AllPairs, use_cuda):
    """
    Return the mean reciprocal rank across a given set of pairs and a given model.

    :param model: siamese network
    :param pairs: dataset of desired pairs to calculate MRR across
    :param use_cuda: bool, whether to run on gpu
    :return: float, mean of reciprocal ranks
    """
    rrs = reciprocal_ranks(model, pairs, use_cuda)
    return rrs.mean()


def reciprocal_ranks(model: Siamese, pairs: AllPairs, use_cuda):
    """
    Return an array of the reciprocal ranks across a given set of pairs and a given model.

    :param model: siamese network
    :param pairs: dataset of desired pairs to calculate RR across
    :param use_cuda: bool, whether to run on gpu
    :return: rrs, ndarray of reciprocal ranks
    """
    pairwise = pairwise_inference_matrix(model, pairs, use_cuda)

    rrs = np.zeros([pairs.n_imitations])
    for i, imitation in enumerate(pairs.imitations):
        # get the column of the pairwise matrix corresponding to this imitation
        pairwise_col = pairwise[i, :]
        # get the index of the correct reference for this imitation
        reference_index = utils.np_index_of(pairs.labels[i, :], 1)
        # get the similarity of the correct reference
        similarity = pairwise_col[reference_index]
        # sort pairwise column descending
        pairwise_col[::-1].sort()
        # find the rank of the similarity
        index = utils.np_index_of(pairwise_col, similarity)
        rank = index + 1
        rrs[i] = 1 / rank

    return rrs


def pairwise_inference_matrix(model: Siamese, pairs_dataset: AllPairs, use_cuda):
    """
    Calculates the pairwise inference matrix for a given model across a set of pairs (typically, all of them).

    :param model: siamese network
    :param pairs_dataset: dataset of desired pairs to calculate pairwise matrix across
    :param use_cuda: bool, whether to run on GPU
    :return: pairwise matrix
    """
    rrs = np.array([])
    pairs = dataloader.DataLoader(pairs_dataset, batch_size=128, num_workers=1)

    bar = Bar("Calculating pairwise inference matrix", max=len(pairs))
    for imitations, references, label in pairs:
        # reshape tensors and push to GPU if necessary
        imitations = imitations.unsqueeze(1)
        references = references.unsqueeze(1)
        if use_cuda:
            imitations = imitations.cuda()
            references = references.cuda()

        output = model(imitations, references)
        # Detach the gradient, move to cpu, and convert to an ndarray
        np_output = output.detach().cpu().numpy()
        rrs = np.concatenate([rrs, np_output])

        bar.next()
    bar.finish()

    # Reshape vector into matrix
    rrs = rrs.reshape([pairs_dataset.n_imitations, pairs_dataset.n_references])
    return rrs


def hard_negative_selection(model: Siamese, pairs: AllPairs, use_cuda):
    """
    Perform hard negative selection to determine negative pairings for fine tuning the network

    :param model: siamese network
    :param pairs: all pairs
    :param use_cuda: bool, whether to run on GPU
    :return: ndarray of reference indexes, indexed by imitation number
    """
    pairwise = pairwise_inference_matrix(model, pairs, use_cuda)

    # zero out all positive examples
    pairwise = pairwise * np.logical_not(pairs.labels)

    # indexes of max in each column
    references = pairwise.argmax(axis=1)
    return references


def convergence(best_mrrs, convergence_threshold):
    return not (len(best_mrrs) <= 2) and np.abs(best_mrrs[len(best_mrrs) - 1] - best_mrrs[len(best_mrrs) - 2]) < convergence_threshold


def siamese_loss(model: Siamese, dataset, objective, use_cuda: bool, batch_size=128):
    """
    Calculates the loss of model over dataset by objective. Optionally run on the GPU.
    :param model: a siamese network
    :param dataset: a dataset of imitation/reference pairs
    :param objective: loss function
    :param use_cuda: whether to run on GPU or not.
    :param batch_size: optional param to set batch_size. Defaults to 128.
    :return:
    """
    data = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    bar = Bar("Calculating loss", max=len(data))
    batch_losses = np.zeros(len(data))
    for i, (left, right, labels) in enumerate(data):
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
        batch_losses[i] = objective(outputs, labels).item()

        bar.next()
    bar.finish()

    return batch_losses


def tower_loss(model: Tower, dataset: TowerData, objective, use_cuda: bool, batch_size=128):
    """
    Calculates the loss of model over dataset by objective. Optionally run on the GPU.
    """
    data = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    bar = Bar("Calculating loss", max=len(data))
    batch_losses = np.zeros(len(data))
    for i, (audio, labels) in enumerate(data):

        # reshape tensors and push to GPU if necessary
        audio = audio.unsqueeze(1)
        if use_cuda:
            audio = audio.cuda()
            labels = labels.cuda()

        # pass a batch through the network
        outputs = model(audio)

        # calculate loss and optimize weights
        labels = labels.long()
        batch_losses[i] = objective(outputs, labels).item()

        bar.next()
    bar.finish()

    return batch_losses


def tower_accuracy(model: Tower, dataset: TowerData, use_cuda: bool, batch_size=128):
    data = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    bar = Bar("Calculating accuracy", max=len(data))
    correct = 0
    total = 0
    for i, (audio, labels) in enumerate(data):

        # reshape tensors and push to GPU if necessary
        audio = audio.unsqueeze(1)
        if use_cuda:
            audio = audio.cuda()
            labels = labels.cuda()

        # pass a batch through the network
        outputs = model(audio)

        labels = labels.long()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        bar.next()
    bar.finish()

    return correct / total
