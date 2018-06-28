import numpy as np
from progress.bar import Bar
from torch.utils.data import dataloader

from utils import utils
from datasets.vocal_sketch import AllPairs
from models.siamese import Siamese


def mean_reciprocal_ranks(model: Siamese, pairs: AllPairs, use_cuda):
    """
    Return the mean reciprocal rank across a given set of pairs and a given model.

    :param model: siamese network
    :param pairs: dataset of desired pairs to calculate confusion matrix across
    :param use_cuda: bool, whether to run on gpu
    :return: float, mean of reciprocal ranks
    """
    rrs = reciprocal_ranks(model, pairs, use_cuda)
    return rrs.mean(), rrs.var()


def reciprocal_ranks(model: Siamese, pairs: AllPairs, use_cuda):
    """
    Return an array of the reciprocal ranks across a given set of pairs and a given model.

    :param model: siamese network
    :param pairs: dataset of desired pairs to calculate confusion matrix across
    :param use_cuda: bool, whether to run on gpu
    :return: rrs, ndarray of reciprocal ranks
    """
    confusion = confusion_matrix(model, pairs, use_cuda)

    rrs = np.zeros([pairs.n_imitations])
    for i, imitation in enumerate(pairs.imitations):
        # get the column of the confusion matrix corresponding to this imitation
        confusion_col = confusion[i, :]
        # get the index of the correct reference for this imitation
        reference_index = utils.np_index_of(pairs.labels[i, :], 1)
        # get the similarity of the correct reference
        similarity = confusion_col[reference_index]
        # sort confusion column descending
        confusion_col[::-1].sort()
        # find the rank of the similarity
        index = utils.np_index_of(confusion_col, similarity)
        rank = index + 1
        rrs[i] = 1 / rank

    return rrs


def confusion_matrix(model: Siamese, pairs_dataset: AllPairs, use_cuda):
    """
    Calculates the confusion matrix for a given model across a set of pairs (typically, all of them).

    :param model: siamese network
    :param pairs_dataset: dataset of desired pairs to calculate confusion matrix across
    :param use_cuda: bool, whether to run on GPU
    :return: confusion matrix
    """
    rrs = np.array([])
    pairs = dataloader.DataLoader(pairs_dataset, batch_size=128, num_workers=1)

    bar = Bar("Calculating confusion matrix", max=len(pairs))
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
    confusion = confusion_matrix(model, pairs, use_cuda)

    # zero out all positive examples
    confusion = confusion * np.logical_not(pairs.labels)

    # indexes of max in each column
    references = confusion.argmax(axis=1)
    return references


def convergence(best_mrrs, convergence_threshold):
    return not (len(best_mrrs) <= 2) and np.abs(best_mrrs[len(best_mrrs) - 1] - best_mrrs[len(best_mrrs) - 2]) < convergence_threshold
