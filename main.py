import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.dataloader as dataloader
import torch.utils.data.sampler as sampler

from datasets import AllPositivesRandomNegatives
from preprocessing import calculate_spectrograms
from siamese import Siamese
from utils import get_best_model, save_model, mrr, get_highest_ranked_original_recording, percent_correct


def train_on_all_data():
    # global parameters
    n_epochs = 30  # 70 in Bongjun's version, 30 in the paper
    model_path = "./models/train_on_all_data/model_{0}"

    # load up the data
    training_data = AllPositivesRandomNegatives()

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese()

    # choose our objective function
    criterion = nn.BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    # train the network using random selection
    suffix = 'random_selection'
    MRRs = train_network(siamese,
                         training_data,
                         criterion, optimizer,
                         n_epochs,
                         model_path, suffix)

    siamese = get_best_model(MRRs, model_path, suffix)
    save_model(model_path, siamese, 'random_selection_final')


def train_with_finetuning():
    # global parameters
    n_epochs = 30  # 70 in Bongjun's version, 30 in the paper
    model_path = "./models/model_{0}"

    # load up the data
    training_data = AllPositivesRandomNegatives()

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese()

    # choose our objective function
    criterion = nn.BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    # train the network using random selection
    suffix = 'random_selection'
    MRRs = train_network(siamese,
                         training_data,
                         criterion, optimizer,
                         n_epochs,
                         model_path, suffix)

    siamese = get_best_model(MRRs, model_path, suffix)
    save_model(model_path, siamese, 'random_selection_final')

    # further train using hard-negative selection until convergence
    n_epochs = 20
    convergence = False
    # same optimizer with a different learning rate
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.0001, weight_decay=.0001, momentum=.9, nesterov=True)
    while not convergence:
        fine_tuning_imitations = []
        fine_tuning_references = []
        fine_tuning_labels = []
        for imitation, r, l in training_data:
            highest_reference, highest_label = get_highest_ranked_original_recording(imitation, training_data, siamese)

            assert highest_reference is not None

            # add the top-ranked pair as an example
            fine_tuning_imitations.append(imitation)
            fine_tuning_references.append(highest_reference)
            fine_tuning_labels.append(highest_label)

        suffix = 'fine_tuned'
        fine_tuning_data = AllPositivesRandomNegatives(fine_tuning_imitations, fine_tuning_references, fine_tuning_labels)
        MRRs = train_network(siamese,
                             fine_tuning_data,
                             criterion, optimizer,
                             n_epochs,
                             model_path, suffix)
        siamese = get_best_model(MRRs, model_path, suffix)
        # TODO: determine when convergence has occurred
    save_model(model_path, siamese, 'fine_tuned_final')

    testing_data = AllPositivesRandomNegatives(*get_data(is_train=False))
    final_mrr = mrr(testing_data, siamese)
    print("Final MRR: {0}".format(final_mrr))


def train_network(model, datasource, objective_function, optimizer, n_epochs, model_save_path, model_save_path_suffix):
    mrrs = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        # if we're using all positives and random negatives, choose new negatives on each epoch
        if isinstance(datasource, AllPositivesRandomNegatives):
            datasource.reselect_negatives()

        train_data = dataloader.DataLoader(datasource, batch_size=128, num_workers=2)
        for i, (left, right, labels) in enumerate(train_data):
            labels = labels.float()
            # clear out the gradients
            optimizer.zero_grad()

            # pass a batch through the network
            left = left.unsqueeze(1)
            right = right.unsqueeze(1)
            outputs = model(left, right)

            # calculate loss and optimize weights
            loss = objective_function(outputs, labels)
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f\tpc: %.3f' % (epoch + 1, i + 1, loss.item(), percent_correct(outputs, labels)))
        save_model(model_save_path, model, "{0}_{1}".format(model_save_path_suffix, epoch))
        mrrs[epoch] = mrr(datasource, model)
    return mrrs


if __name__ == "__main__":
    calculate_spectrograms()
    train_on_all_data()
