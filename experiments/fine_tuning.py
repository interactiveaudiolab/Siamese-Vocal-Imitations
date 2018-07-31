import logging
import traceback

import numpy as np
import torch
from torch.nn import BCELoss

import experiments.siamese
from data_files.generics import Datafiles
from data_sets.pair import FineTuned, AllPairs
from models.siamese import Siamese
from data_partitions.generics import Partitions
from utils import utils as utilities, experimentation as experimentation, training as training, graphing as graphing


def train(use_cuda, data: Datafiles, use_dropout, validate_every, data_split, regenerate_splits, regenerate_weights, use_cached_baseline=False,
          minimum_passes=0):
    logger = logging.getLogger('logger')
    # get the baseline network
    if use_cached_baseline:
        siamese = Siamese(dropout=use_dropout)
        if use_cuda:
            siamese = siamese.cuda()
        utilities.load_model(siamese, './model_output/random_selection/model_best')
    else:
        siamese = experiments.siamese.train(use_cuda, data, use_dropout, validate_every, data_split, regenerate_splits, regenerate_weights)

    model_path = './model_output/fine_tuned/model_{0}_{1}'

    partitions = Partitions(data, data_split)
    fine_tuning_data = FineTuned(partitions.train)
    training_pairs = AllPairs(partitions.train)
    validation_pairs = AllPairs(partitions.val)
    testing_pairs = AllPairs(partitions.test)

    criterion = BCELoss()

    # further train using hard-negative selection until convergence
    n_epochs = 20
    best_validation_mrrs = []
    best_training_mrrs = []

    convergence_threshold = .01  # TODO: figure out a real number for this
    fine_tuning_pass = 0

    # same optimizer with a different learning rate
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.001, weight_decay=.0001, momentum=.9, nesterov=True)
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
            models = training.train_siamese_network(siamese, fine_tuning_data, criterion, optimizer, n_epochs, use_cuda)
            for epoch, (model, training_batch_losses) in enumerate(models):
                utilities.save_model(model, model_path.format(fine_tuning_pass, epoch))

                validation_batch_losses = experimentation.siamese_loss(model, validation_pairs, criterion, use_cuda)
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

                graphing.mrr_per_epoch(training_mrrs, validation_mrrs, 'Fine Tuning')
                graphing.loss_per_epoch(training_losses, validation_losses, 'Fine Tuning')

            utilities.load_model(siamese, model_path.format(fine_tuning_pass, np.argmax(validation_mrrs[fine_tuning_pass * n_epochs:])))
            utilities.save_model(siamese, model_path.format(fine_tuning_pass, 'best'))
            best_validation_mrrs.append(np.max(validation_mrrs[fine_tuning_pass * n_epochs:]))
            best_training_mrrs.append(np.max(training_mrrs[fine_tuning_pass * n_epochs:]))
            graphing.mrr_per_epoch(best_training_mrrs, best_validation_mrrs, 'Fine Tuning', title='Best MRR vs. Fine Tuning Pass', xlabel="fine tuning pass")

            fine_tuning_pass += 1

        utilities.load_model(siamese, model_path.format(np.argmax(best_validation_mrrs), 'best'))
        utilities.save_model(siamese, model_path.format('best', 'best'))
        utilities.save_model(siamese, './output/{0}/fine_tuned'.format(utilities.get_trial_number()))

        rrs = experimentation.reciprocal_ranks(siamese, testing_pairs, use_cuda)
        logger.info("Results from best model generated after tine-tuning, evaluated on test data:")
        utilities.log_final_stats(rrs)
    except Exception as e:
        utilities.save_model(siamese, model_path.format('crash', 'backup'))
        print("Exception occurred while training: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)
