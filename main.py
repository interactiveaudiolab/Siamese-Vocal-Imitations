import argparse
import logging
import traceback

import numpy as np
import torch
import torch.optim
from torch.nn import BCELoss, CrossEntropyLoss

# MUST COME FIRST
# noinspection PyUnresolvedReferences
from utils import matplotlib_backend_hack

import utils.experimentation as experimentation
import utils.graphing as graphing
import utils.utils as utilities
import utils.training as training

from datafiles.voxforge import Voxforge
from datafiles.urban_sound_8k import UrbanSound8K
from datafiles.vocal_sketch import VocalSketch

from datasets.voxforge import All
from datasets.urban_sound_8k import UrbanSound10FCV
from datasets.vocal_sketch import AllPositivesRandomNegatives, AllPairs, FineTuned

from models.siamese import Siamese
from models.transfer_learning import RightTower, LeftTower


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

    try:
        logger.info("Copying weights from left tower...")
        left_tower = LeftTower()
        utilities.load_model(left_tower, './model_output/left_tower/model_final')
        training.copy_weights(siamese, left_tower)
    except FileNotFoundError:
        logger.error("Could not find a pre-trained left tower model. Skipping transfer learning for this branch.")
    try:
        logger.info("Copying weights from right tower...")
        right_tower = RightTower()
        utilities.load_model(right_tower, './model_output/right_tower/model_final')
        training.copy_weights(siamese, right_tower)
    except FileNotFoundError:
        logger.error("Could not find a pre-trained right tower model. Skipping transfer learning for this branch.")

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
        utilities.save_model(siamese, model_path.format('crash_backup'))
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
        utilities.save_model(siamese, model_path.format('crash', 'backup'))
        print("Exception occurred while training: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)


def left_tower_transfer_learning(use_cuda, data: Voxforge):
    logger = logging.getLogger('logger')
    model_path = './model_output/left_tower/model_{0}'

    n_epochs = 50
    model = LeftTower()
    if use_cuda:
        model.cuda()

    loss = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, weight_decay=.001, momentum=.9, nesterov=True)
    dataset = All(data)
    try:
        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []
        logger.info("Training left tower....")
        models = training.train_tower(model, dataset, loss, optimizer, n_epochs, use_cuda)
        for epoch, (model, training_batch_losses) in enumerate(models):
            utilities.save_model(model, model_path.format(epoch))

            dataset.validation_mode()
            validation_batch_losses = experimentation.tower_loss(model, dataset, loss, use_cuda)
            dataset.training_mode()

            training_loss = training_batch_losses.mean()
            validation_loss = validation_batch_losses.mean()
            logger.info("Loss at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_loss, validation_loss))

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            graphing.loss_per_epoch(training_losses, validation_losses, title='Loss vs. Epoch (TL, Left Tower)')

            training_accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
            dataset.validation_mode()
            validation_accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
            dataset.training_mode()

            training_accuracies.append(training_accuracy)
            validation_accuracies.append(validation_accuracy)
            logger.info("Accuracy at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_accuracy, validation_accuracy))

            graphing.accuracy_per_epoch(training_accuracies, validation_accuracies, 'Accuracy vs. Epoch (TL, Left Tower)')

        dataset.validation_mode()
        accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
        dataset.training_mode()
        logger.info("Final validation accuracy = {0}".format(accuracy))
        utilities.save_model(model, "./output/{0}/left_tower".format(utilities.get_trial_number()))
        utilities.save_model(model, model_path.format('final'))
    except Exception as e:
        utilities.save_model(model, model_path.format('crash_backup'))
        print("Exception occurred while training left tower: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)


def right_tower_transfer_learning(use_cuda, data: UrbanSound8K):
    logger = logging.getLogger('logger')
    short_model_path = './model_output/right_tower/model_{0}'
    model_path = short_model_path + '_{1}'

    n_epochs = 50
    model = RightTower()
    if use_cuda:
        model.cuda()

    loss = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, weight_decay=.001, momentum=.9, nesterov=True)
    dataset = UrbanSound10FCV(data)
    try:
        fold_accuracies = np.zeros(dataset.n_folds)
        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []
        for fold in range(dataset.n_folds):
            logger.info("Training right tower, validating on fold {0}...".format(fold))
            dataset.set_fold(fold)
            models = training.train_tower(model, dataset, loss, optimizer, n_epochs, use_cuda)
            for epoch, (model, training_batch_losses) in enumerate(models):
                utilities.save_model(model, model_path.format(fold, epoch))

                dataset.validation_mode()
                validation_batch_losses = experimentation.tower_loss(model, dataset, loss, use_cuda)
                dataset.training_mode()

                training_loss = training_batch_losses.mean()
                validation_loss = validation_batch_losses.mean()
                logger.info("Loss at fold {0}, epoch {1}:\n\ttrn = {2}\n\tval = {3}".format(fold, epoch, training_loss, validation_loss))

                training_losses.append(training_loss)
                validation_losses.append(validation_loss)

                graphing.loss_per_epoch(training_losses, validation_losses, title='Loss vs. Epoch (TL, Right Tower)')

                training_accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
                dataset.validation_mode()
                validation_accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
                dataset.training_mode()

                training_accuracies.append(training_accuracy)
                validation_accuracies.append(validation_accuracy)

                logger.info("Accuracy at fold {0}, epoch {1}:\n\ttrn = {2}\n\tval = {3}".format(fold, epoch, training_accuracy, validation_accuracy))

                graphing.accuracy_per_epoch(training_accuracies, validation_accuracies, 'Accuracy vs. Epoch (TL, Right Tower)')

            accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
            fold_accuracies[fold] = accuracy
            dataset.validation_mode()
            logger.info("Validation accuracy on fold {0} = {1}".format(fold, accuracy))
            dataset.training_mode()
            utilities.save_model(model, "./output/{0}/right_tower".format(utilities.get_trial_number()))
            utilities.save_model(model, short_model_path.format('final'))

        logger.info("Average accuracy across all folds = {0}".format(np.mean(fold_accuracies)))
    except Exception as e:
        utilities.save_model(model, short_model_path.format('crash_backup'))
        print("Exception occurred while training right tower: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)


def main(cli_args=None):
    utilities.update_trial_number()
    utilities.create_output_directory()

    logger = logging.getLogger('logger')
    parser = argparse.ArgumentParser()
    utilities.configure_parser(parser)
    utilities.configure_logger(logger)
    if cli_args is None:
        cli_args = parser.parse_args()

    logger.info('Beginning trial #{0}...'.format(utilities.get_trial_number()))

    # log all CLI args
    for key in vars(cli_args):
        logger.debug("\t{0} = {1}".format(key, vars(cli_args)[key]))

    try:
        if cli_args.transfer_learning in ['left', 'imitation', 'both']:
            voxforge = Voxforge(recalculate_spectrograms=cli_args.spectrograms)
            left_tower_transfer_learning(cli_args.cuda, voxforge)

        if cli_args.transfer_learning in ['right', 'reference', 'both']:
            urban_sound = UrbanSound8K(recalculate_spectrograms=cli_args.spectrograms)
            right_tower_transfer_learning(cli_args.cuda, urban_sound)

        vocal_sketch = VocalSketch(*cli_args.partitions, recalculate_spectrograms=cli_args.spectrograms)
        if cli_args.random_only:
            train_random_selection(cli_args.cuda, vocal_sketch, cli_args.dropout, cli_args.no_normalization)
        else:
            train_fine_tuning(cli_args.cuda, vocal_sketch, cli_args.dropout, cli_args.no_normalization, use_cached_baseline=cli_args.cache_baseline,
                              minimum_passes=cli_args.fine_tuning_passes)
        cli_args.trials -= 1
        if cli_args.trials > 0:
            main(cli_args)
    except Exception as e:
        logger.critical("Unhandled exception: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)


if __name__ == "__main__":
    main()
