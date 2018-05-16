import numpy as np
import torch

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


def mrr(imitations, original_recordings, siamese):
    total_rr = 0
    for imitation, original_recording in zip(imitations, original_recordings):
        total_rr += rr(imitation, original_recording, original_recordings, siamese)
    return total_rr / len(imitations)


def rr(imitation, reference, original_recordings, siamese):
    assert reference in original_recordings

    rankings = {}
    for original_recording in original_recordings:
        rankings[original_recording] = siamese(imitation, original_recording)

    rankings = [(r, rankings[r]) for r in rankings.keys()]
    rankings.sort(key=lambda x: x[1])
    for i, original_recording, ranking in enumerate(rankings):
        if original_recording == reference:
            return 1 / i


def get_highest_ranked_original_recording(imitation, original_recordings, siamese):
    highest_ranking = 0
    highest_ranked_example = None
    highest_ranked_label = None
    # find the highest ranked original recording
    for original_recording in original_recordings:
        output = siamese(imitation, original_recording)
        if output > highest_ranking:
            highest_ranking = output
            highest_ranked_example = original_recording
            highest_ranked_label = 0  # TODO: get actual label
    return highest_ranked_example, highest_ranked_label


def percent_correct(outputs, labels):
    correct = num_correct(outputs, labels)
    total = labels.shape[0]
    return correct / total


def num_correct(outputs, labels):
    return torch.sum(torch.round(outputs) == labels).item()