import numpy as np
import torch
from progress.bar import Bar
from torch.utils.data import dataloader

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


def mean_reciprocal_ranks(data, references, model, use_cuda):
    rrs = reciprocal_ranks(data, model, references, use_cuda)
    return rrs.mean()


def reciprocal_ranks(data, model, references, use_cuda):
    train_data = dataloader.DataLoader(data, batch_size=1, num_workers=1)
    rrs = np.array([])
    bar = Bar("Calculating reciprocal rank", max=len(train_data))
    for imitation, reference, label in train_data:
        if label.data[0]:
            rrs = np.append(rrs, reciprocal_rank(imitation, reference, references, model, use_cuda))
        bar.next()
    bar.finish()
    return rrs


def reciprocal_rank(imitation, reference, references, model, use_cuda):
    references = dataloader.DataLoader(references, batch_size=1, num_workers=1)
    # reshape tensors and push to GPU if necessary
    imitation = imitation.unsqueeze(1)
    reference = reference.unsqueeze(1)
    if use_cuda:
        imitation = imitation.cuda()
        reference = reference.cuda()

    match_output = None
    rankings = []
    for r in references:
        # reshape tensor and push to GPU if necessary
        r = r.unsqueeze(1)
        if use_cuda:
            r = r.cuda()

        # perform inference
        output = model(imitation, r)

        # if this one is the matching reference, hold on to that value
        if torch.equal(r, reference):
            match_output = output
        rankings.append(output.tolist()[0])  # get single value out of tensor

    rankings.sort(reverse=True)

    # rr = 1 / rank of matching reference
    return 1 / (rankings.index(match_output) + 1)  # rank = index + 1


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
