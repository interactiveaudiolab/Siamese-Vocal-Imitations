import logging
import pickle
import traceback

import numpy as np
import torch
from torch.nn import BCELoss

from data_files.generics import Datafiles
from data_partitions.generics import Partitions
from data_partitions.pair import PairPartition
from data_partitions.triplet import TripletPartition
from data_sets.pair import AllPairs
from data_sets.triplet import AllPositivesRandomNegatives
from models.triplet import Triplet
from models.siamese import Siamese
from utils import utils as utilities, training as training, experimentation as experimentation, graphing as graphing
from utils.obj import DataSplit, TrainingProgress
from utils.utils import initialize_weights


def train(n_epochs: int, use_cuda,
          data: Datafiles, data_split: DataSplit, regenerate_splits: bool, n_categories: int, validate_every: int,
          use_dropout: bool, regenerate_weights: bool,
          optimizer_name: str, lr: float, wd: float, momentum: bool):
    logger = logging.getLogger('logger')

    model_path = "./model_output/triplet/model_{0}"
    no_test = True

    triplet_partitions = Partitions(data, data_split, TripletPartition, n_train_val_categories=n_categories, no_test=no_test,
                                    regenerate_splits=regenerate_splits)
    training_data = AllPositivesRandomNegatives(triplet_partitions.train)
    validation_data = AllPositivesRandomNegatives(triplet_partitions.val)

    if validate_every > 0:
        pair_partitions = Partitions(data, data_split, PairPartition, regenerate_splits=False, no_test=no_test)
        training_pairs = AllPairs(pair_partitions.train)
        search_length = training_pairs.n_references
        validation_pairs = AllPairs(pair_partitions.val)
        testing_pairs = AllPairs(pair_partitions.test) if not no_test else None
    else:
        training_pairs = None
        validation_pairs = None
        testing_pairs = None
        search_length = None

    siamese = initialize_weights(Siamese(dropout=use_dropout), regenerate_weights, use_cuda)
    network = Triplet(dropout=use_dropout)
    network.load_siamese(siamese)

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
        progress = TrainingProgress()
        models = training.train_triplet_network(network, training_data, criterion, optimizer, n_epochs, use_cuda)
        for epoch, (model, training_batch_losses) in enumerate(models):
            utilities.save_model(model, model_path.format(epoch))

            should_validate = validate_every != 0 and epoch % validate_every == 0

            training_loss = training_batch_losses.mean()
            if should_validate:
                validation_batch_losses = experimentation.triplet_loss(model, validation_data, criterion, use_cuda, batch_size=64)
                validation_loss = validation_batch_losses.mean()
                logger.info("Loss at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_loss, validation_loss))

                logger.debug("Calculating MRRs...")
                training_mrr, training_rank = experimentation.mean_reciprocal_ranks(network.siamese, training_pairs, use_cuda)
                val_mrr, val_rank = experimentation.mean_reciprocal_ranks(network.siamese, validation_pairs, use_cuda)
                logger.info("MRRs at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_mrr, val_mrr))
                logger.info("Mean ranks at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_rank, val_rank))

                progress.add_mrr(train=training_mrr, val=val_mrr)
                progress.add_rank(train=training_rank, val=val_rank)
                progress.add_loss(train=training_loss, val=validation_loss)
            else:
                logger.info("Loss at epoch {0}:\n\ttrn = {1}".format(epoch, training_loss))
                progress.add_mrr(train=np.nan, val=np.nan)
                progress.add_rank(train=np.nan, val=np.nan)
                progress.add_loss(train=training_loss, val=np.nan)

            progress.graph("Triplet", search_length)

        # load weights from best model if we validated throughout
        if validate_every > 0:
            network = network.train()
            utilities.load_model(network, model_path.format(np.argmax(progress.val_mrr)))
        # otherwise just save most recent model
        utilities.save_model(network, model_path.format('best'))
        utilities.save_model(network, './output/{0}/triplet'.format(utilities.get_trial_number()))
        if not no_test:
            logger.info("Results from best model generated during random-selection training, evaluated on test data:")
            rrs = experimentation.reciprocal_ranks(network, testing_pairs, use_cuda)
            utilities.log_final_stats(rrs)

        training_result = TrainingProgress()
        train_correlation, val_correlation = training_result.pearson()
        logger.info("Correlations between loss and MRR:\n\ttrn = {0}\n\tval = {1}".format(train_correlation, val_correlation))
        with open("./output/{0}/triplet.pickle".format(utilities.get_trial_number()), 'w+b') as f:
            pickle.dump(training_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        return network
    except Exception as e:
        utilities.save_model(network, model_path.format('crash_backup'))
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)
