import numpy as np
import torch
from progress.bar import Bar
from torch import Tensor
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


def mean_reciprocal_ranks(model, all_pairs, use_cuda):
    rrs = confusion_matrix(model, all_pairs, use_cuda)
    rrs = rrs.reshape([all_pairs.n_imitations, all_pairs.n_references])

    a = np.zeros([all_pairs.n_imitations])
    for i, imitation in enumerate(all_pairs.imitations):
        output_col = rrs[i, :]
        label_col = all_pairs.labels[i, :]
        match_location = np.where(label_col == 1)[0][0]
        match = output_col[match_location]
        output_col[::-1].sort()
        index = np.where(output_col == match)[0][0]
        a[i] = 1 / (index + 1)

    return a.mean()


def confusion_matrix(model, all_pairs, use_cuda):
    rrs = np.array([])
    all_pairs = dataloader.DataLoader(all_pairs, batch_size=128, num_workers=1)

    bar = Bar("Calculating confusion matrix", max=len(all_pairs))
    for imitations, references, label in all_pairs:
        # reshape tensors and push to GPU if necessary
        imitations = imitations.unsqueeze(1)
        references = references.unsqueeze(1)
        if use_cuda:
            imitations = imitations.cuda()
            references = references.cuda()

        output = model(imitations, references)
        rrs = np.concatenate([rrs, output.detach().cpu().numpy()])

        bar.next()
    bar.finish()

    return rrs


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
