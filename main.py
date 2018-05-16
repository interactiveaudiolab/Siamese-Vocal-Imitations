import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.dataloader as dataloader

from datasets import FileVocalImitations, DynamicVocalImitations
from siamese import Siamese


def main():
    # global parameters
    n_epochs = 30  # 70 in Bongjun's version, 30 in the paper
    model_path = "./model_{0}"

    # load up the data
    imitations = []  # TODO: get imitation data
    original_recordings = []  # TODO: get original recordings
    train_data = dataloader.DataLoader(FileVocalImitations(), batch_size=128, num_workers=2, shuffle=True)

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese()

    # choose our objective function
    criterion = nn.BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    # train the network using random selection
    train_network(siamese, criterion, optimizer, n_epochs, train_data)
    save_model(model_path, siamese, 'random_selection')

    # further train using hard-negative selection until convergence
    n_epochs = 20
    convergence = False
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.0001, weight_decay=.0001, momentum=.9, nesterov=True)
    while not convergence:
        left = []
        right = []
        labels = []
        for imitation in imitations:
            highest_ranked_example, highest_ranked_label = get_highest_ranked_original_recording(imitation, original_recordings, siamese)

            assert highest_ranked_example is not None

            # add the top-ranked pair as an example
            left.append(imitation)
            right.append(highest_ranked_example)
            labels.append(highest_ranked_label)

        dynamic_dataset = dataloader.DataLoader(DynamicVocalImitations(left, right, labels), batch_size=128)
        train_network(siamese, criterion, optimizer, n_epochs, dynamic_dataset)
        # TODO: find out when the network is converging and end the loop

    save_model(model_path, siamese, 'fine_tuned')

    # evaluate the network
    test_data = dataloader.DataLoader(FileVocalImitations(is_train=False), batch_size=128, num_workers=2, shuffle=True)
    total_correct = 0
    total = 0
    for i, (left, right, labels) in enumerate(test_data):
        outputs = siamese(left, right)
        total_correct += num_correct(outputs, labels)
        total += labels.shape[0]
        print("Cumulative percent correct on validation data: {0}%".format(100 * total_correct / total_correct))


def train_network(model, objective_function, optimizer, n_epochs, train_data):
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
