import logging
import traceback

import numpy as np
import torch
from torch.nn import BCELoss

from datafiles.generics import SiameseDatafile
from datasets.siamese import AllPositivesRandomNegatives, AllPairs
from models.siamese import Siamese
from utils import utils as utilities, training as training, experimentation as experimentation, graphing as graphing


def train(use_cuda, data: SiameseDatafile, use_dropout, use_normalization):
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
        models = training.train_siamese_network(siamese, training_data, criterion, optimizer, n_epochs, use_cuda)
        for epoch, (model, training_batch_losses) in enumerate(models):
            utilities.save_model(model, model_path.format(epoch))

            validation_batch_losses = experimentation.siamese_loss(model, validation_pairs, criterion, use_cuda)
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

            graphing.mrr_per_epoch(training_mrrs, validation_mrrs, "Random Selection")
            graphing.loss_per_epoch(training_losses, validation_losses, "Random Selection")

        # get and save best model TODO: should this be by training or by validation?
        utilities.load_model(siamese, model_path.format(np.argmax(validation_mrrs)))
        utilities.save_model(siamese, model_path.format('best'))
        utilities.save_model(siamese, './output/{0}/random_selection'.format(utilities.get_trial_number()))

        logger.info("Results from best model generated during random-selection training, evaluated on test data:")
        rrs = experimentation.reciprocal_ranks(siamese, testing_pairs, use_cuda)
        utilities.log_final_stats(rrs)
        return siamese
    except Exception as e:
        utilities.save_model(siamese, model_path.format('crash_backup'))
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)
