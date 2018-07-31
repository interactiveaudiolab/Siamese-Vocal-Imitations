import logging
import pickle
import traceback

import numpy as np
import torch
from torch.nn import BCELoss

from data_files.generics import Datafiles
from data_partitions.siamese import PairPartition
from data_sets.pair import AllPositivesRandomNegatives, AllPairs
from models.siamese import Siamese
from data_partitions.generics import Partitions
from utils import utils as utilities, training as training, experimentation as experimentation, graphing as graphing
from utils.obj import TrainingResult


def train(use_cuda, data: Datafiles, use_dropout, validate_every, data_split, regenerate_splits, regenerate_weights, optimizer_name, lr, wd, momentum,
          n_epochs):
    logger = logging.getLogger('logger')

    model_path = "./model_output/siamese/model_{0}"

    partitions = Partitions(data, data_split, PairPartition, regenerate_splits=regenerate_splits)
    training_data = AllPositivesRandomNegatives(partitions.train)
    training_pairs = AllPairs(partitions.train)
    search_length = training_pairs.n_references
    validation_pairs = AllPairs(partitions.val)
    testing_pairs = AllPairs(partitions.test)

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese(dropout=use_dropout)
    starting_weights_path = model_path.format("starting_weights")

    if regenerate_weights:
        logger.debug("Saving initial weights/biases at {0}...".format(starting_weights_path))
        utilities.save_model(siamese, starting_weights_path)
    else:
        try:
            logger.debug("Loading initial weights/biases from {0}...".format(starting_weights_path))
            utilities.load_model(siamese, starting_weights_path, use_cuda)
        except FileNotFoundError:
            logger.debug("Saving initial weights/biases at {0}...".format(starting_weights_path))
            utilities.save_model(siamese, starting_weights_path)
    if use_cuda:
        siamese = siamese.cuda()

    criterion = BCELoss()

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(siamese.parameters(), lr=lr, weight_decay=wd, momentum=.9 if momentum else 0, nesterov=momentum)  # TODO: separate params
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(siamese.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(siamese.parameters(), lr=lr, weight_decay=wd, momentum=.9 if momentum else 0)
    else:
        raise ValueError("No optimizer found with name {0}".format(optimizer_name))

    try:
        logger.info("Training using random selection...")

        training_mrrs = []
        validation_mrrs = []
        training_losses = []
        validation_losses = []
        training_ranks = []
        validation_ranks = []

        models = training.train_siamese_network(siamese, training_data, criterion, optimizer, n_epochs, use_cuda)

        # logger.debug("Calculating MRRs...")
        # training_mrr = experimentation.mean_reciprocal_ranks(siamese, training_pairs, use_cuda)
        # val_mrr = experimentation.mean_reciprocal_ranks(siamese, validation_pairs, use_cuda)
        # logger.info("MRRs at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(-1, training_mrr, val_mrr))
        # training_mrrs.append(training_mrr)
        # validation_mrrs.append(val_mrr)

        for epoch, (model, training_batch_losses) in enumerate(models):
            utilities.save_model(model, model_path.format(epoch))

            should_validate = validate_every != 0 and epoch % validate_every == 0

            training_loss = training_batch_losses.mean()
            training_losses.append(training_loss)
            if should_validate:
                validation_batch_losses = experimentation.siamese_loss(model, validation_pairs, criterion, use_cuda)
                validation_loss = validation_batch_losses.mean()
                validation_losses.append(validation_loss)
                logger.info("Loss at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_loss, validation_loss))

                logger.debug("Calculating MRRs...")
                training_mrr, training_rank = experimentation.mean_reciprocal_ranks(model, training_pairs, use_cuda)
                val_mrr, val_rank = experimentation.mean_reciprocal_ranks(model, validation_pairs, use_cuda)
                logger.info("MRRs at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_mrr, val_mrr))
                logger.info("Mean ranks at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_rank, val_rank))

                training_mrrs.append(training_mrr)
                validation_mrrs.append(val_mrr)

                training_ranks.append(training_rank)
                validation_ranks.append(val_rank)

                graphing.mrr_per_epoch(training_mrrs, validation_mrrs, "Siamese", n_categories=search_length)
            else:
                logger.info("Loss at epoch {0}:\n\ttrn = {1}".format(epoch, training_loss))
                validation_losses.append(np.nan)
                training_mrrs.append(np.nan)
                validation_mrrs.append(np.nan)

            graphing.loss_per_epoch(training_losses, validation_losses, "Siamese")

        # load weights from best model if we validated throughout
        if validate_every > 0:
            siamese = siamese.train()
            utilities.load_model(siamese, model_path.format(np.argmax(validation_mrrs)))
        # otherwise just save most recent model
        utilities.save_model(siamese, model_path.format('best'))
        utilities.save_model(siamese, './output/{0}/siamese'.format(utilities.get_trial_number()))

        # logger.info("Results from best model generated during random-selection training, evaluated on test data:")
        # rrs = experimentation.reciprocal_ranks(siamese, testing_pairs, use_cuda)
        # utilities.log_final_stats(rrs)

        training_result = TrainingResult(training_mrrs, training_ranks, training_losses, validation_mrrs, validation_ranks, validation_losses)
        train_correlation, val_correlation = training_result.pearson()
        logger.info("Correlations between loss and MRR:\n\ttrn = {0}\n\tval = {1}".format(train_correlation, val_correlation))
        with open("./output/{0}/siamese.pickle".format(utilities.get_trial_number()), 'w+b') as f:
            pickle.dump(training_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        return siamese
    except Exception as e:
        utilities.save_model(siamese, model_path.format('crash_backup'))
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)
