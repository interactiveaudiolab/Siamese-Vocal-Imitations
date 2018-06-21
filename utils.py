import os

import numpy as np
import torch
from progress.bar import Bar
from torch.utils.data import dataloader

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
    load_model(model, best_model, base_path, path_suffix)
    return model


def load_model(model, best_epoch, base_path, path_suffix):
    model.load_state_dict(torch.load(base_path.format("{0}_{1}".format(path_suffix, best_epoch))))


def save_model(base_path, model, path_suffix):
    torch.save(model.state_dict(), base_path.format(path_suffix))


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


def load_npy(name):
    return np.load(os.environ['SIAMESE_DATA_DIR'] + "/npy/" + name)


def save_npy(array, suffix, type):
    np.save(os.environ['SIAMESE_DATA_DIR'] + "/npy/" + suffix, np.array(array).astype(type))


def prindent(str, n_indent):
    p_str = ''
    for i in range(n_indent):
        p_str += '\t'
    p_str += str
    print(p_str)


def print_final_stats(rrs):
    prindent("mean: {0}".format(rrs.mean()), 1)
    prindent("stddev: {0}".format(rrs.std()), 1)
    prindent("min: {0}".format(rrs.min()), 1)
    prindent("max: {0}".format(rrs.max()), 1)
