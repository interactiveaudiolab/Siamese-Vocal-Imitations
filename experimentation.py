import numpy as np
import torch
from progress.bar import Bar
from torch.utils.data import dataloader

import utils
from siamese import Siamese


def get_best_model(MRRs, base_path, path_suffix):
    """
    Get the best model based on mean reciprocal rank.

    :param MRRs: epoch-indexed array of mean reciprocal ranks
    :param base_path:
    :param path_suffix:
    :return:
    """
    best_model = np.argmax(MRRs)
    model = Siamese()
    utils.load_model_from_epoch(model, best_model, base_path, path_suffix)
    return model


def mean_reciprocal_ranks(model, pairs, use_cuda):
    """
    Return the mean reciprocal rank across a given set of pairs and a given model.

    :param model: siamese network
    :param pairs: dataset of desired pairs to calculate confusion matrix across
    :param use_cuda: bool, whether to run on gpu
    :return: float, mean of reciprocal ranks
    """
    return reciprocal_ranks(model, pairs, use_cuda).mean()


def reciprocal_ranks(model, pairs, use_cuda):
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


def confusion_matrix(model, pairs_dataset, use_cuda):
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


def hard_negative_selection(imitation, reference, references, model, use_cuda):
    highest_ranking = 0
    highest_ranked_example = None

    # reshape tensors and push to GPU if necessary
    imitation = imitation.unsqueeze(1)
    reference = reference.unsqueeze(1)
    if use_cuda:
        imitation = imitation.cuda()
        reference = reference.cuda()

    for r in references:
        # reshape tensor and push to GPU if necessary
        r = r.unsqueeze(1)
        if use_cuda:
            r = r.cuda()
        output = model(imitation, r)
        if output > highest_ranking and not torch.equal(r, reference):
            highest_ranking = output
            highest_ranked_example = r
    return highest_ranked_example
