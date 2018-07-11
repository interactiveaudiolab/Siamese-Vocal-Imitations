import logging
import traceback

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from datafiles.urban_sound_8k import UrbanSound8K
from datafiles.voxforge import Voxforge
from datasets.urban_sound_8k import UrbanSound10FCV
from datasets.voxforge import All
from models.transfer_learning import LeftTower, RightTower
from utils import training as training, utils as utilities, experimentation as experimentation, graphing as graphing


def train_left(use_cuda, data: Voxforge):
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

            graphing.loss_per_epoch(training_losses, validation_losses, 'TL, Left Tower')

            training_accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
            dataset.validation_mode()
            validation_accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
            dataset.training_mode()

            training_accuracies.append(training_accuracy)
            validation_accuracies.append(validation_accuracy)
            logger.info("Accuracy at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_accuracy, validation_accuracy))

            graphing.accuracy_per_epoch(training_accuracies, validation_accuracies, 'TL, Left Tower')

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


def train_right(use_cuda, data: UrbanSound8K):
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

                graphing.loss_per_epoch(training_losses, validation_losses, 'TL, Right Tower')

                training_accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
                dataset.validation_mode()
                validation_accuracy = experimentation.tower_accuracy(model, dataset, use_cuda)
                dataset.training_mode()

                training_accuracies.append(training_accuracy)
                validation_accuracies.append(validation_accuracy)

                logger.info("Accuracy at fold {0}, epoch {1}:\n\ttrn = {2}\n\tval = {3}".format(fold, epoch, training_accuracy, validation_accuracy))

                graphing.accuracy_per_epoch(training_accuracies, validation_accuracies, 'TL, Right Tower')

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