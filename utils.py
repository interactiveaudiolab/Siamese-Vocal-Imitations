import numpy as np
import torch
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


def mrr(data, model):
    train_data = dataloader.DataLoader(data, batch_size=1, num_workers=2)
    total_rr = 0
    for imitation, reference, label in train_data:
        total_rr += rr(imitation, reference, train_data, model)
    return total_rr / len(train_data)


def rr(imitation, reference, data, model):
    rankings = {}
    for i, r, l in data:
        rankings[r] = model(imitation, r)

    rankings = [(r, rankings[r]) for r in rankings.keys()]
    rankings.sort(key=lambda x: x[1])
    for i, original_recording, ranking in enumerate(rankings):
        if original_recording == reference:
            return 1 / i
    # Should not get here
    assert False


def get_highest_ranked_original_recording(imitation, data, siamese):
    highest_ranking = 0
    highest_ranked_example = None
    highest_ranked_label = None
    # find the highest ranked original recording
    for i, r, l in data:
        output = siamese(imitation, r)
        if output > highest_ranking:
            highest_ranking = output
            highest_ranked_example = r
            highest_ranked_label = 0  # TODO: get actual label
    return highest_ranked_example, highest_ranked_label


def percent_correct(outputs, labels):
    correct = num_correct(outputs, labels)
    total = labels.shape[0]
    return correct / total


def num_correct(outputs, labels):
    return torch.sum(torch.round(outputs) == labels).item()


def load_npy(name):
    return np.load("./data/npy/" + name)


def save_npy(array, suffix, type):
    np.save("./data/npy/" + suffix, np.array(array).astype(type))