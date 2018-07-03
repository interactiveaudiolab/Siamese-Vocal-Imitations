import argparse
import logging
import traceback

import numpy as np
import torch
import torch.optim
from progress.bar import Bar
from torch.nn import BCELoss
from torch.utils.data.dataloader import DataLoader

# noinspection PyUnresolvedReferences
from utils import matplotlib_backend_hack
from datafiles.vocal_sketch_files import VocalSketch
from datasets.vocal_sketch_data import AllPositivesRandomNegatives, AllPairs, FineTuned
import utils.experimentation as experimentation
import utils.graphing as graphing
import utils.preprocessing as preprocessing
import utils.utils as utilities
from models.siamese import Siamese


def train_random_selection(use_cuda, data: VocalSketch, use_dropout, use_normalization):
    logger = logging.getLogger('logger')

    n_epochs = 100
    model_path = "./model_output/random_selection/model_{0}"

    training_data = AllPositivesRandomNegatives(data.train)
    training_pairs = AllPairs(data.train)
    validation_pairs = AllPairs(data.val)
    testing_pairs = AllPairs(data.test)

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese(dropout=use_dropout, normalization=use_normalization)
    if use_cuda:
        siamese = siamese.cuda()

    criterion = BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    try:
        logger.info("Training using random selection...")
        training_mrrs = np.zeros(n_epochs)
        validation_mrrs = np.zeros(n_epochs)
        training_losses = np.zeros(n_epochs)
        training_loss_var = np.zeros(n_epochs)
        validation_losses = np.zeros(n_epochs)
        models = train_network(siamese, training_data, criterion, optimizer, n_epochs, use_cuda)
        for epoch, (model, training_batch_losses) in enumerate(models):
            utilities.save_model(model, model_path.format(epoch))

            validation_batch_losses = experimentation.loss(model, validation_pairs, criterion, use_cuda)
            training_loss = training_batch_losses.mean()
            validation_loss = validation_batch_losses.mean()
            logger.info("Loss at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_loss, validation_loss))

            training_losses[epoch] = training_loss
            training_loss_var[epoch] = training_batch_losses.var()
            validation_losses[epoch] = validation_loss

            logger.debug("Calculating MRRs...")
            training_mrr = experimentation.mean_reciprocal_ranks(model, training_pairs, use_cuda)
            val_mrr = experimentation.mean_reciprocal_ranks(model, validation_pairs, use_cuda)
            logger.info("MRRs at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_mrr, val_mrr))

            training_mrrs[epoch] = training_mrr
            validation_mrrs[epoch] = val_mrr

            graphing.mrr_per_epoch(training_mrrs, validation_mrrs, title="MRR vs. Epoch (Random Selection)")
            graphing.loss_per_epoch(training_losses, validation_losses, title='Loss vs. Epoch (Random Selection)')

        # get and save best model TODO: should this be by training or by validation?
        utilities.load_model(siamese, model_path.format(np.argmax(validation_mrrs)))
        utilities.save_model(siamese, model_path.format('best'))
        utilities.save_model(siamese, './output/{0}/random_selection'.format(utilities.get_trial_number()))

        logger.info("Results from best model generated during random-selection training, evaluated on test data:")
        rrs = experimentation.reciprocal_ranks(siamese, testing_pairs, use_cuda)
        utilities.log_final_stats(rrs)
        return siamese
    except Exception as e:
        utilities.save_model(siamese, model_path)
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)


def train_fine_tuning(use_cuda, data: VocalSketch, use_dropout, use_normalization, use_cached_baseline=False, minimum_passes=0):
    logger = logging.getLogger('logger')
    # get the baseline network
    if use_cached_baseline:
        siamese = Siamese(dropout=use_dropout, normalization=use_normalization)
        if use_cuda:
            siamese = siamese.cuda()
        utilities.load_model(siamese, './model_output/random_selection/model_best')
    else:
        siamese = train_random_selection(use_cuda, data, use_dropout, use_normalization)

    model_path = './model_output/fine_tuned/model_{0}_{1}'

    fine_tuning_data = FineTuned(data.train)
    training_pairs = AllPairs(data.train)
    validation_pairs = AllPairs(data.val)
    testing_pairs = AllPairs(data.test)

    criterion = BCELoss()

    # further train using hard-negative selection until convergence
    n_epochs = 20
    best_validation_mrrs = []
    best_training_mrrs = []

    convergence_threshold = .01  # TODO: figure out a real number for this
    fine_tuning_pass = 0

    # same optimizer with a different learning rate
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.0001, weight_decay=.0001, momentum=.9, nesterov=True)
    try:
        # fine tune until convergence
        logger.info("Fine tuning model, minimum # of passes = {0}".format(minimum_passes))
        training_mrrs = []
        validation_mrrs = []
        training_losses = []
        training_loss_var = []
        validation_losses = []
        while not experimentation.convergence(best_validation_mrrs, convergence_threshold) or fine_tuning_pass < minimum_passes:
            fine_tuning_data.reset()
            logger.debug("Performing hard negative selection...")
            references = experimentation.hard_negative_selection(siamese, training_pairs, use_cuda)
            fine_tuning_data.add_negatives(references)

            logger.info("Beginning fine tuning pass {0}...".format(fine_tuning_pass))
            models = train_network(siamese, fine_tuning_data, criterion, optimizer, n_epochs, use_cuda)
            for epoch, (model, training_batch_losses) in enumerate(models):
                utilities.save_model(model, model_path.format(fine_tuning_pass, epoch))

                validation_batch_losses = experimentation.loss(model, validation_pairs, criterion, use_cuda)
                training_loss = training_batch_losses.mean()
                validation_loss = validation_batch_losses.mean()
                logger.info("Loss at pass {0}, epoch {1}:\n\ttrn = {2}\n\tval = {3}".format(fine_tuning_pass, epoch, training_loss, validation_loss))

                training_losses.append(training_loss)
                training_loss_var.append(training_batch_losses.var())
                validation_losses.append(validation_loss)

                logger.debug("Calculating MRRs...")
                training_mrr = experimentation.mean_reciprocal_ranks(model, training_pairs, use_cuda)
                val_mrr = experimentation.mean_reciprocal_ranks(model, validation_pairs, use_cuda)
                logger.info("MRRs at pass {0}, epoch {1}:\n\ttrn = {2}\n\tval = {3}".format(fine_tuning_pass, epoch, training_mrr, val_mrr))

                training_mrrs.append(training_mrr)
                validation_mrrs.append(val_mrr)

                graphing.mrr_per_epoch(training_mrrs, validation_mrrs, title='MRR vs. Epoch (Fine Tuning)')
                graphing.loss_per_epoch(training_losses, validation_losses, title='Loss vs. Epoch (Fine Tuning)')

            utilities.load_model(siamese, model_path.format(fine_tuning_pass, np.argmax(validation_mrrs[fine_tuning_pass * n_epochs:])))
            utilities.save_model(siamese, model_path.format(fine_tuning_pass, 'best'))
            best_validation_mrrs.append(np.max(validation_mrrs[fine_tuning_pass * n_epochs:]))
            best_training_mrrs.append(np.max(training_mrrs[fine_tuning_pass * n_epochs:]))
            graphing.mrr_per_epoch(best_training_mrrs, best_validation_mrrs, title='Best MRR vs. Fine Tuning Pass (Fine Tuning)', xlabel="fine tuning pass")

            fine_tuning_pass += 1

        utilities.load_model(siamese, model_path.format(np.argmax(best_validation_mrrs), 'best'))
        utilities.save_model(siamese, model_path.format('best', 'best'))
        utilities.save_model(siamese, './output/{0}/fine_tuned'.format(utilities.get_trial_number()))

        rrs = experimentation.reciprocal_ranks(siamese, testing_pairs, use_cuda)
        logger.info("Results from best model generated after tine-tuning, evaluated on test data:")
        utilities.log_final_stats(rrs)
    except Exception as e:
        utilities.save_model(siamese, model_path)
        print("Exception occurred while training: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)


def train_network(model, data, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # if we're using all positives and random negatives, choose new negatives on each epoch
        if isinstance(data, AllPositivesRandomNegatives):
            data.reselect_negatives()

        train_data = DataLoader(data, batch_size=batch_size, num_workers=1)
        bar = Bar("Training epoch {0}".format(epoch), max=len(train_data))
        batch_losses = np.zeros(len(train_data))
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
            batch_losses[i] = loss.item()

            bar.next()
        bar.finish()

        yield model, batch_losses


def main(cli_args=None):
    utilities.update_trial_number()
    utilities.create_output_directory()

    logger = logging.getLogger('logger')
    parser = argparse.ArgumentParser()
    utilities.configure_parser(parser)
    utilities.configure_logger(logger)
    if cli_args is None:
        cli_args = parser.parse_args()

    n_trials = cli_args.trials
    logger.info('Beginning trial #{0}...'.format(utilities.get_trial_number()))

    # log all CLI args
    arg_dict = vars(cli_args)
    for key in arg_dict:
        logger.debug("\t{0} = {1}".format(key, arg_dict[key]))

    try:
        if cli_args.spectrograms:
            preprocessing.load_data_set()
        vocal_sketch = VocalSketch(*cli_args.partitions)
        if cli_args.random_only:
            train_random_selection(cli_args.cuda, vocal_sketch, cli_args.dropout, cli_args.normalization)
        else:
            train_fine_tuning(cli_args.cuda, vocal_sketch, cli_args.dropout, cli_args.normalization, use_cached_baseline=cli_args.cache_baseline,
                              minimum_passes=cli_args.fine_tuning_passes)
        n_trials -= 1
        cli_args.trials = n_trials
        if n_trials > 0:
            main(cli_args)
    except Exception as e:
        logger.critical("Unhandled exception: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)


if __name__ == "__main__":
    main()
