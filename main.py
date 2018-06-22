import sys
import datetime
import traceback

import numpy as np
import torch
from progress.bar import Bar
from torch.nn import BCELoss
import torch.optim
from torch.utils.data.dataloader import DataLoader

import data_utils
import experimentation
import datasets
from siamese import Siamese


def train_random_selection(use_cuda):
    # global parameters
    n_epochs = 70  # 70 in Bongjun's version, 30 in the paper
    model_path = "./models/train_on_all_data/model_{0}"

    training_data = datasets.AllPositivesRandomNegatives()
    positive_pairs = datasets.AllPositivePairs()

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese()
    if use_cuda:
        siamese = siamese.cuda()

    criterion = BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    # train the network using random selection
    suffix = 'random_selection'

    try:
        MRRs = train_network(siamese,
                             training_data, positive_pairs,
                             criterion, optimizer,
                             n_epochs,
                             model_path, suffix, use_cuda, calculate_mrr=True)
        rrs = experimentation.reciprocal_ranks(siamese, positive_pairs, use_cuda)
    except Exception as e:
        data_utils.save_model(model_path, siamese, 'crash_backup_{0}'.format(datetime.datetime.now()))
        print("Exception occurred while training: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)

    print("Results from training on all positives and random negatives:")
    data_utils.print_final_stats(rrs)

    return siamese

    # siamese = get_best_model(MRRs, model_path, suffix)
    # for mrr_result in MRRs:
    #     print("mrr_result: {0}".format(mrr_result))
    # save_model(model_path, siamese, 'random_selection_final')


def train_fine_tuning(use_cuda, use_cached_baseline=False):
    # get the baseline network
    if use_cached_baseline:
        siamese = Siamese()
        if use_cuda:
            siamese = siamese.cuda()
        data_utils.load_model('./models/train_on_all_data/model_{0}', siamese, 'random_selection_final')
    else:
        siamese = train_random_selection(use_cuda)

    # global parameters
    model_path = './models/fine_tuned/model_{0}'

    positive_pairs_data = datasets.AllPositivePairs()
    positive_pairs = DataLoader(positive_pairs_data, batch_size=1, num_workers=1)
    references_data = datasets.AllReferences()
    references = DataLoader(references_data, batch_size=1, num_workers=1)

    criterion = BCELoss()
    try:
        # further train using hard-negative selection until convergence
        n_epochs = 20
        best_mrrs = []
        convergence_threshold = .01  # TODO: figure out a real number for this

        # same optimizer with a different learning rate
        optimizer = torch.optim.SGD(siamese.parameters(), lr=.0001, weight_decay=.0001, momentum=.9, nesterov=True)

        # fine tune until convergence
        while not convergence(best_mrrs, convergence_threshold):
            fine_tuning_data = datasets.FineTuned()

            bar = Bar("Running hard-negative selection...", max=len(positive_pairs))
            for imitation, reference in positive_pairs:
                highest_reference = experimentation.get_highest_ranked_negative_reference(imitation, reference, references, siamese, use_cuda)
                fine_tuning_data.add_negative(imitation, highest_reference)
                bar.next()
            bar.finish()

            suffix = 'fine_tuned'
            epoch_mrrs = train_network(siamese,
                                       fine_tuning_data, references,
                                       criterion, optimizer,
                                       n_epochs,
                                       model_path, suffix, use_cuda)
            siamese = experimentation.get_best_model(epoch_mrrs, model_path, suffix)
            best_mrrs.append(np.max(epoch_mrrs))
            rrs = experimentation.reciprocal_ranks(fine_tuning_data, siamese, references, use_cuda)

        data_utils.save_model(model_path, siamese, 'fine_tuned_final')
        final_mrr = experimentation.mean_reciprocal_ranks(fine_tuning_data, references, siamese, use_cuda)
    except Exception as e:
        data_utils.save_model(model_path, siamese, 'crash_backup_{0}'.format(datetime.datetime.now()))
        print("Exception occurred while training: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)

    print("Results from training after fine-tuning:")
    data_utils.print_final_stats(rrs)

    print("Final MRR: {0}".format(final_mrr))


def convergence(best_mrrs, convergence_threshold):
    return not (len(best_mrrs) <= 2) and np.abs(best_mrrs[len(best_mrrs) - 1] - best_mrrs[len(best_mrrs) - 2]) < convergence_threshold


def train_network(model, data, positive_pairs, objective, optimizer, n_epochs, model_save_path, model_save_path_suffix, use_cuda, calculate_mrr=True):
    mrrs = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        # if we're using all positives and random negatives, choose new negatives on each epoch
        if isinstance(data, datasets.AllPositivesRandomNegatives):
            data.reselect_negatives()

        train_data = DataLoader(data, batch_size=32, num_workers=1)
        bar = Bar("Training epoch {0}".format(epoch), max=len(train_data))
        for i, (left, right, labels) in enumerate(train_data):
            # clear out the gradients
            optimizer.zero_grad()

            # TODO: make them floats at the source
            labels = labels.float()

            # reshape tensors and push to GPU if necessary
            left = left.unsqueeze(1)
            right = right.unsqueeze(1)
            if use_cuda:
                left = left.cuda()
                right = right.cuda()
                labels = labels.cuda()

            # pass a batch through the network
            outputs = model(left, right)

            # calculate loss and optimize weights
            loss = objective(outputs, labels)
            loss.backward()
            optimizer.step()

            bar.next()
        bar.finish()

        data_utils.save_model(model_save_path, model, "{0}_{1}".format(model_save_path_suffix, epoch))
        if calculate_mrr:
            mrr_result = experimentation.mean_reciprocal_ranks(model, positive_pairs, use_cuda)
            print("MRR at epoch {0} = {1}".format(epoch, mrr_result))
            mrrs[epoch] = mrr_result
    return mrrs


if __name__ == "__main__":
    use_cuda_arg = sys.argv[1] if len(sys.argv) > 1 else False
    print("CUDA {0}...".format("enabled" if use_cuda_arg else "disabled"))

    # calculate_spectrograms()
    train_random_selection(use_cuda_arg)
    # train_fine_tuning(use_cuda_arg, use_cached_baseline=False)
