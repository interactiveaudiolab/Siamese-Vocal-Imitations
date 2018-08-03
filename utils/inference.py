import logging

import numpy as np
from torch.utils.data import dataloader, DataLoader

from data_sets.pair import AllPairs
from models.triplet import Triplet
from models.siamese import Siamese
from utils import utils
from utils.progress_bar import Bar


def mean_reciprocal_ranks(model: Siamese, pairs: AllPairs, use_cuda):
    """
    Return the mean reciprocal rank across a given set of pairs and a given model.

    :param model: siamese network
    :param pairs: dataset of desired pairs to calculate MRR across
    :param use_cuda: bool, whether to run on gpu
    :return: float, mean of reciprocal ranks
    """
    logger = logging.getLogger('logger')
    logger.debug("Calculating MRRs...")

    rrs, ranks = reciprocal_ranks(model, pairs, use_cuda)
    return rrs.mean(), ranks.mean()


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
    ranks = np.zeros([pairs.n_imitations])
    for i, imitation in enumerate(pairs.imitations):
        # get the column of the pairwise matrix corresponding to this imitation
        pairwise_col = pairwise[i, :]
        # get the index of the correct canonical reference for this imitation
        reference_index = utils.np_index_of(pairs.canonical_labels[i, :], 1)
        # get the similarity of the correct reference
        similarity = pairwise_col[reference_index]
        # sort pairwise column descending
        pairwise_col[::-1].sort()
        # find the rank of the similarity
        index = utils.np_index_of(pairwise_col, similarity)
        rank = index + 1
        rrs[i] = 1 / rank
        ranks[i] = rank

    return rrs, ranks


def pairwise_inference_matrix(model: Siamese, pairs_dataset: AllPairs, use_cuda):
    """
    Calculates the pairwise inference matrix for a given model across a set of pairs (typically, all of them).

    :param model: siamese network
    :param pairs_dataset: dataset of desired pairs to calculate pairwise matrix across
    :param use_cuda: bool, whether to run on GPU
    :return: pairwise matrix
    """
    rrs = np.array([])
    pairs = dataloader.DataLoader(pairs_dataset, batch_size=128, num_workers=4)
    model = model.eval()
    bar = Bar("Calculating pairwise inference matrix", max=len(pairs))
    for imitations, references, label in pairs:

        label = label.float()
        imitations = imitations.float()
        references = references.float()

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
    model = model.eval()
    dataset.epoch_handler()

    data = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    bar = Bar("Calculating loss", max=len(data))
    batch_losses = np.zeros(len(data))
    for i, (left, right, labels) in enumerate(data):
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
        batch_losses[i] = objective(outputs, labels).item()

        bar.next()
    bar.finish()

    return batch_losses


def triplet_loss(model: Triplet, dataset, objective, use_cuda: bool, batch_size=128):
    """
    Calculates the loss of model over dataset by objective. Optionally run on the GPU.
    :param model: a siamese network
    :param dataset: a dataset of imitation/reference pairs
    :param objective: loss function
    :param use_cuda: whether to run on GPU or not.
    :param batch_size: optional param to set batch_size. Defaults to 128.
    :return:
    """
    model = model.eval()
    dataset.epoch_handler()

    data = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    bar = Bar("Calculating loss", max=len(data))
    batch_losses = np.zeros(len(data))
    for i, triplet in enumerate(data):
        # clear out the gradients
        triplet = [tensor.float() for tensor in triplet]

        # reshape tensors and push to GPU if necessary
        triplet = [tensor.unsqueeze(1) for tensor in triplet[:3]] + [triplet[3]]
        if use_cuda:
            triplet = [tensor.cuda() for tensor in triplet]

        # pass a batch through the network
        outputs = model(*triplet[:3])

        # calculate loss and optimize weights
        batch_losses[i] = objective(outputs, triplet[3]).item()

        bar.next()
    bar.finish()

    return batch_losses
