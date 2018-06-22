import numpy as np
import torch
from progress.bar import Bar
from torch.utils.data import dataloader

import datasets
from data_utils import load_model_from_epoch
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
    load_model_from_epoch(model, best_model, base_path, path_suffix)
    return model


def mean_reciprocal_ranks(model, positive_pairs, use_cuda):
    rrs = reciprocal_ranks(model, positive_pairs, use_cuda)
    return rrs.mean()


def reciprocal_ranks(model, positive_pairs, use_cuda):
    rrs = np.array([])
    all_pairs = datasets.OneImitationAllReferences()
    bar = Bar("Calculating reciprocal rank", max=len(positive_pairs))
    positive_pairs = dataloader.DataLoader(positive_pairs, batch_size=1, num_workers=1)
    for imitation, reference in positive_pairs:
        all_pairs.set_imitation(imitation)
        rr = reciprocal_rank(model, reference, all_pairs, use_cuda)
        rrs = np.append(rrs, rr)
        bar.next()
    bar.finish()
    return rrs


def reciprocal_rank(model, correct_reference, all_pairs, use_cuda):
    # reshape tensors and push to GPU if necessary
    correct_reference = correct_reference.unsqueeze(1)
    if use_cuda:
        correct_reference = correct_reference.cuda()

    match_output = None
    rankings = []
    all_pairs = dataloader.DataLoader(all_pairs, batch_size=32, num_workers=1)
    for imitation, reference in all_pairs:
        # reshape tensors and push to GPU if necessary
        # do not unsqueeze the imitation tensor - it already has the right shape
        reference = reference.unsqueeze(1)
        if use_cuda:
            imitation = imitation.cuda()
            reference = reference.cuda()

        # # perform inference
        # print(imitation.shape)
        # print(reference.shape)
        # exit(-1)
        output = model(imitation, reference)

        # if this one is the matching reference, hold on to that value
        for i, r in enumerate(reference):
            r = r.unsqueeze(1)
            if torch.equal(r, correct_reference):
                match_output = output.tolist()[i]

        rankings = np.concatenate([rankings, output.tolist()])

    # sort descending
    rankings[::-1].sort()

    # rr = 1 / rank of matching reference
    return 1 / (np.where(rankings == match_output)[0][0] + 1)  # rank = index + 1


def get_highest_ranked_negative_reference(imitation, reference, references, model, use_cuda):
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


def percent_correct(outputs, labels):
    correct = num_correct(outputs, labels)
    total = labels.shape[0]
    return correct / total


def num_correct(outputs, labels):
    return torch.sum(torch.round(outputs) == labels).item()
