import logging
import traceback

import numpy as np
import torch
from torch.nn import BCELoss

from data_files.generics import Datafiles
from data_partitions.generics import Partitions
from data_partitions.siamese import PairPartition
from data_partitions.triplet import TripletPartition
from data_sets.pair import AllPairs
from data_sets.triplet import AllPositivesRandomNegatives
from models.bisiamese import Bisiamese
from utils import utils as utilities, training as training, experimentation as experimentation, graphing as graphing
from utils.obj import DataSplit


def train(n_epochs: int, use_cuda,
          data: Datafiles, data_split: DataSplit, regenerate_splits: bool, n_categories: int, validate_every: int,
          use_dropout: bool, regenerate_weights: bool,
          optimizer_name: str, lr: float, wd: float, momentum: bool):
    logger = logging.getLogger('logger')

    model_path = "./model_output/triplet/model_{0}"

    triplet_partitions = Partitions(data, data_split, TripletPartition, n_train_val_categories=n_categories, no_test=True, regenerate_splits=regenerate_splits)
    training_data = AllPositivesRandomNegatives(triplet_partitions.train)
    validation_data = AllPositivesRandomNegatives(triplet_partitions.val)

    if validate_every > 0:
        no_test = True
        pair_partitions = Partitions(data, data_split, PairPartition, regenerate_splits=False, no_test=no_test)
        training_pairs = AllPairs(pair_partitions.train)
        validation_pairs = AllPairs(pair_partitions.val)
        testing_pairs = AllPairs(pair_partitions.test) if not no_test else None
    else:
        training_pairs = None
        validation_pairs = None
        testing_pairs = None

    # get a bisiamese network, see Siamese class for architecture
    network = Bisiamese(dropout=use_dropout)
    starting_weights_path = model_path.format("starting_weights")

    if regenerate_weights:
        logger.debug("Saving initial weights/biases at {0}...".format(starting_weights_path))
        utilities.save_model(network, starting_weights_path)
    else:
        try:
            logger.debug("Loading initial weights/biases from {0}...".format(starting_weights_path))
            utilities.load_model(network, starting_weights_path, use_cuda)
        except FileNotFoundError:
            logger.debug("Saving initial weights/biases at {0}...".format(starting_weights_path))
            utilities.save_model(network, starting_weights_path)
    if use_cuda:
        network = network.cuda()

    criterion = BCELoss()

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, weight_decay=wd, momentum=.9 if momentum else 0, nesterov=momentum)  # TODO: separate params
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(), lr=lr, weight_decay=wd, momentum=.9 if momentum else 0)
    else:
        raise ValueError("No optimizer found with name {0}".format(optimizer_name))

    try:
        logger.info("Training triplet network...")

        training_mrrs = []
        validation_mrrs = []
        training_losses = []
        validation_losses = []

        models = training.train_bisiamese_network(network, training_data, criterion, optimizer, n_epochs, use_cuda)

        if validate_every > 0:
            logger.debug("Calculating initial MRRs...")
            training_mrr = experimentation.mean_reciprocal_ranks(network.siamese, training_pairs, use_cuda)
            val_mrr = experimentation.mean_reciprocal_ranks(network.siamese, validation_pairs, use_cuda)
            logger.info("MRRs at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(-1, training_mrr, val_mrr))

            training_mrrs.append(training_mrr)
            validation_mrrs.append(val_mrr)

        for epoch, (model, training_batch_losses) in enumerate(models):
            utilities.save_model(model, model_path.format(epoch))

            should_validate = validate_every != 0 and epoch % validate_every == 0

            training_loss = training_batch_losses.mean()
            training_losses.append(training_loss)
            if should_validate:
                validation_batch_losses = experimentation.bisiamese_loss(model, validation_data, criterion, use_cuda, batch_size=64)
                validation_loss = validation_batch_losses.mean()
                validation_losses.append(validation_loss)
                logger.info("Loss at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_loss, validation_loss))

                logger.debug("Calculating MRRs...")
                training_mrr = experimentation.mean_reciprocal_ranks(network.siamese, training_pairs, use_cuda)
                val_mrr = experimentation.mean_reciprocal_ranks(network.siamese, validation_pairs, use_cuda)
                logger.info("MRRs at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_mrr, val_mrr))

                training_mrrs.append(training_mrr)
                validation_mrrs.append(val_mrr)

                graphing.mrr_per_epoch(training_mrrs, validation_mrrs, "Random Selection", n_categories=n_categories)
            else:
                logger.info("Loss at epoch {0}:\n\ttrn = {1}".format(epoch, training_loss))
                validation_losses.append(np.nan)
                training_mrrs.append(np.nan)
                validation_mrrs.append(np.nan)

            graphing.loss_per_epoch(training_losses, validation_losses, "Random Selection")

        # load weights from best model if we validated throughout
        if validate_every > 0:
            network = network.train()
            utilities.load_model(network, model_path.format(np.argmax(validation_mrrs)))
        # otherwise just save most recent model
        utilities.save_model(network, model_path.format('best'))
        utilities.save_model(network, './output/{0}/random_selection'.format(utilities.get_trial_number()))

        logger.info("Results from best model generated during random-selection training, evaluated on test data:")
        rrs = experimentation.reciprocal_ranks(network, testing_pairs, use_cuda)
        utilities.log_final_stats(rrs)
        return network
    except Exception as e:
        utilities.save_model(network, model_path.format('crash_backup'))
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)
