import argparse
import datetime
import logging
import traceback

import numpy as np
import torch
import torch.optim
from progress.bar import Bar
from torch.nn import BCELoss
from torch.utils.data.dataloader import DataLoader

import utils
import datasets
import experimentation
import preprocessing
from siamese import Siamese


def train_random_selection(use_cuda, limit=None):
    logger = logging.getLogger('logger')
    # global parameters
    n_epochs = 70  # 70 in Bongjun's version, 30 in the paper
    model_path = "./models/random_selection/model_{0}"

    training_data = datasets.AllPositivesRandomNegatives(limit)
    all_pairs = datasets.AllPairs(limit)

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese()
    if use_cuda:
        siamese = siamese.cuda()

    criterion = BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    try:
        logging.info("Training using random selection...")
        mrrs = np.zeros(n_epochs)
        models = train_network(siamese, training_data, criterion, optimizer, n_epochs, use_cuda)
        for epoch, model in enumerate(models):
            mrr = experimentation.mean_reciprocal_ranks(model, all_pairs, use_cuda)
            utils.save_model(model, model_path.format(epoch))
            logger.info("MRR at epoch {0} = {1}".format(epoch, mrr))
            mrrs[epoch] = mrr

        # get best model
        utils.load_model(siamese, model_path.format(np.argmax(mrrs)))
        utils.save_model(siamese, model_path.format('best'))

        rrs = experimentation.reciprocal_ranks(siamese, all_pairs, use_cuda)
        logger.info("Results from best model generated during random-selection training:")
        utils.log_final_stats(rrs)
        return siamese
    except Exception as e:
        utils.save_model(siamese, model_path)
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)


def train_fine_tuning(use_cuda, use_cached_baseline=False, limit=None):
    logger = logging.getLogger('logger')
    # get the baseline network
    if use_cached_baseline:
        siamese = Siamese()
        if use_cuda:
            siamese = siamese.cuda()
        utils.load_model(siamese, './models/random_selection/model_best')
    else:
        siamese = train_random_selection(use_cuda, limit)

    # global parameters
    model_path = './models/fine_tuned/model_{0}_{1}'

    all_pairs = datasets.AllPairs(limit)
    fine_tuning_data = datasets.FineTuned()

    criterion = BCELoss()

    # further train using hard-negative selection until convergence
    n_epochs = 20
    best_mrrs = []
    convergence_threshold = .01  # TODO: figure out a real number for this
    fine_tuning_pass = 0

    # same optimizer with a different learning rate
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.0001, weight_decay=.0001, momentum=.9, nesterov=True)
    try:
        # fine tune until convergence
        while not convergence(best_mrrs, convergence_threshold):
            references = experimentation.hard_negative_selection(siamese, all_pairs, use_cuda)
            fine_tuning_data.add_negatives(references)

            logging.info("Beginning fine tuning pass {0}...".format(fine_tuning_pass))
            mrrs = np.zeros(n_epochs)
            models = train_network(siamese, fine_tuning_data, criterion, optimizer, n_epochs, use_cuda)
            for epoch, model in enumerate(models):
                mrr = experimentation.mean_reciprocal_ranks(model, all_pairs, use_cuda)
                utils.save_model(model, model_path)
                logger.info("MRR at epoch {0}, fine tuning pass {1} = {2}".format(epoch, fine_tuning_pass, mrr))
                mrrs[epoch] = mrr

            utils.load_model(siamese, model_path.format(fine_tuning_pass, np.argmax(mrrs)))
            utils.save_model(siamese, model_path.format(fine_tuning_pass, 'best'))
            best_mrrs.append(np.max(mrrs))
            fine_tuning_pass += 1

        utils.load_model(siamese, model_path.format(np.argmax(best_mrrs), 'best'))
        utils.save_model(siamese, model_path.format('best', 'best'))
        rrs = experimentation.reciprocal_ranks(siamese, all_pairs, use_cuda)
        print("Results from training after fine-tuning:")
        utils.log_final_stats(rrs)
    except Exception as e:
        utils.save_model(siamese, model_path)
        print("Exception occurred while training: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)


def convergence(best_mrrs, convergence_threshold):
    return not (len(best_mrrs) <= 2) and np.abs(best_mrrs[len(best_mrrs) - 1] - best_mrrs[len(best_mrrs) - 2]) < convergence_threshold


def train_network(model, data, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # if we're using all positives and random negatives, choose new negatives on each epoch
        if isinstance(data, datasets.AllPositivesRandomNegatives):
            data.reselect_negatives()

        train_data = DataLoader(data, batch_size=batch_size, num_workers=1)
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

        yield model


def main():
    # set up logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    # handlers and formatter
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('siamese.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s \t %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--limit', default=None, type=int, help='Optional limit on the size of the dataset, useful for debugging')
    parser.add_argument('-c', '--cuda', action='store_const', const=True, default=False, help='Whether to enable calculation on the GPU through CUDA or not')
    parser.add_argument('-s', '--spectrograms', action='store_const', const=True, default=False, help='Whether to calculate spectrograms or not')
    parser.add_argument('-b', '--cache_baseline', action='store_const', const=True, default=False,
                        help='Whether to use a cached version of the baseline model')
    args = parser.parse_args()

    logger.info('Beginning experiment...')
    logger.info("CUDA {0}...".format("enabled" if args.cuda else "disabled"))
    if args.limit:
        logger.info("Limiting to {0} imitations/references".format(args.limit))

    try:
        if args.spectrograms:
            preprocessing.calculate_spectrograms()
        train_fine_tuning(args.cuda, use_cached_baseline=args.cache_baseline)
        # train_random_selection(args.cuda, limit=args.limit)
    except Exception as e:
        logger.critical("Unhandled exception: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)


if __name__ == "__main__":
    main()
