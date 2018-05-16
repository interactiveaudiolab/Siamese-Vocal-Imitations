import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.dataloader as dataloader

from datasets import VocalImitations, get_data
from siamese import Siamese
from utils import get_best_model, save_model, mrr, get_highest_ranked_original_recording, percent_correct


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


if __name__ == "__main__":
    main()
