import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.dataloader as dataloader

from datasets import VocalImitations, get_data
from siamese import Siamese


def main():
    # global parameters
    n_epochs = 30  # 70 in Bongjun's version, 30 in the paper
    model_path = "./models/model_{0}"

    # load up the data
    all_imitations, all_references, all_labels = get_data()

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese()

    # choose our objective function
    criterion = nn.BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    # train the network using random selection
    suffix = 'random_selection'
    MRRs = train_network(siamese, all_imitations, all_references, all_labels, criterion, optimizer, n_epochs, model_path, suffix)

    siamese = get_best_model(MRRs, model_path, suffix)
    save_model(model_path, siamese, 'random_selection_final')

    # further train using hard-negative selection until convergence
    n_epochs = 20
    convergence = False
    # same optimizer with a different learning rate
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.0001, weight_decay=.0001, momentum=.9, nesterov=True)
    while not convergence:
        left = []
        right = []
        all_labels = []
        for imitation in all_imitations:
            highest_ranked_example, highest_ranked_label = get_highest_ranked_original_recording(imitation, all_references, siamese)

            assert highest_ranked_example is not None

            # add the top-ranked pair as an example
            left.append(imitation)
            right.append(highest_ranked_example)
            all_labels.append(highest_ranked_label)

        suffix = 'fine_tuned'
        MRRs = train_network(siamese, left, right, all_labels, criterion, optimizer, n_epochs, model_path, suffix)
        siamese = get_best_model(MRRs, model_path, suffix)
        # TODO: determine when convergence has occurred
    save_model(model_path, siamese, 'fine_tuned_final')

    all_imitations, all_references, all_labels = get_data(is_train=False)
    final_mrr = mrr(all_imitations, all_references, siamese)
    print("Final MRR: {0}".format(final_mrr))


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


def train_network(model, imitations, references, labels, objective_function, optimizer, n_epochs, model_save_path, model_save_path_suffix):
    train_data = dataloader.DataLoader(VocalImitations(imitations, references, labels), batch_size=128, num_workers=2, shuffle=True)
    mrr = list(range(n_epochs))
    for epoch in range(n_epochs):
        for i, (left, right, labels) in enumerate(train_data):
            labels = labels.float()
            # clear out the gradients
            optimizer.zero_grad()

            # pass a batch through the network
            outputs = model(left, right)

            # calculate loss and optimize weights
            loss = objective_function(outputs, labels)
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f\tpc: %.3f' % (epoch + 1, i + 1, loss.item(), percent_correct(outputs, labels)))
        save_model(model_save_path, model, "{0}_{1}".format(model_save_path_suffix, epoch))
        mrr[epoch] = mrr(imitations, references, model)
    return mrr


def evaluate_network(model, imitations, references):
    return mrr(imitations, references, model)


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


if __name__ == "__main__":
    main()
