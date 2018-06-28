import argparse
import logging
import traceback

import numpy as np
import torch
import torch.optim
from progress.bar import Bar
from torch.nn import BCELoss
from torch.utils.data.dataloader import DataLoader

import datafiles.vocal_sketch_files as vocal_sketch_files
import datasets.vocal_sketch_data as vocal_sketch_data
import utils.experimentation
import utils.graphing
import utils.preprocessing
import utils.utils
from models.siamese import Siamese


def train_random_selection(use_cuda, data: vocal_sketch_files.VocalSketch):
    logger = logging.getLogger('logger')

    n_epochs = 70  # 70 in Bongjun's version, 30 in the paper
    model_path = "./model_output/random_selection/model_{0}"

    training_data = vocal_sketch_data.AllPositivesRandomNegatives(data.train)
    training_pairs = vocal_sketch_data.AllPairs(data.train)
    validation_pairs = vocal_sketch_data.AllPairs(data.val)
    testing_pairs = vocal_sketch_data.AllPairs(data.test)

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese()
    if use_cuda:
        siamese = siamese.cuda()

    criterion = BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    try:
        logger.info("Training using random selection...")
        training_mrrs = np.zeros(n_epochs)
        training_mrr_vars = np.zeros(n_epochs)
        validation_mrrs = np.zeros(n_epochs)
        validation_mrr_vars = np.zeros(n_epochs)
        losses = np.zeros(n_epochs)
        loss_var = np.zeros(n_epochs)
        models = train_network(siamese, training_data, criterion, optimizer, n_epochs, use_cuda)
        for epoch, (model, loss) in enumerate(models):
            utils.utils.save_model(model, model_path.format(epoch))

            logger.debug("Calculating MRRs...")
            training_mrr, training_mrr_var = utils.experimentation.mean_reciprocal_ranks(model, training_pairs, use_cuda)
            val_mrr, val_mrr_var = utils.experimentation.mean_reciprocal_ranks(model, validation_pairs, use_cuda)
            logger.info("MRRs at epoch {0}:\n\ttrn = {1}\n\tval = {2}".format(epoch, training_mrr, val_mrr))
            logger.info("Loss at epoch {0} = {1}".format(epoch, loss.mean()))

            training_mrrs[epoch] = training_mrr
            validation_mrrs[epoch] = val_mrr

            training_mrr_vars[epoch] = training_mrr_var
            validation_mrr_vars[epoch] = val_mrr_var

            losses[epoch] = loss.mean()
            loss_var[epoch] = loss.var()

            utils.graphing.mrr_per_epoch(training_mrrs, validation_mrrs, training_mrr_vars, validation_mrr_vars, title="MRR vs. Epoch (Random Selection)")
            utils.graphing.loss_per_epoch(losses, loss_var)

        # get and save best model TODO: should this be by training or by validation?
        utils.utils.load_model(siamese, model_path.format(np.argmax(validation_mrrs)))
        utils.utils.save_model(siamese, model_path.format('best'))

        logger.info("Results from best model generated during random-selection training, evaluated on test data:")
        rrs = utils.experimentation.reciprocal_ranks(siamese, testing_pairs, use_cuda)
        utils.utils.log_final_stats(rrs)
        utils.graphing.loss_per_epoch(losses, loss_var)
        return siamese
    except Exception as e:
        utils.utils.save_model(siamese, model_path)
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)


def train_fine_tuning(use_cuda, data: vocal_sketch_files.VocalSketch, use_cached_baseline=False, minimum_passes=None):
    logger = logging.getLogger('logger')
    # get the baseline network
    if use_cached_baseline:
        siamese = Siamese()
        if use_cuda:
            siamese = siamese.cuda()
        utils.utils.load_model(siamese, './model_output/random_selection/model_best')
    else:
        siamese = train_random_selection(use_cuda, data)

    model_path = './model_output/fine_tuned/model_{0}_{1}'

    fine_tuning_data = vocal_sketch_data.FineTuned(data.train)
    training_pairs = vocal_sketch_data.AllPairs(data.train)
    validation_pairs = vocal_sketch_data.AllPairs(data.val)
    testing_pairs = vocal_sketch_data.AllPairs(data.test)

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
        while not utils.experimentation.convergence(best_validation_mrrs, convergence_threshold) or \
                (minimum_passes is not None and fine_tuning_pass < minimum_passes):
            fine_tuning_data.reset()
            logger.debug("Performing hard negative selection...")
            references = utils.experimentation.hard_negative_selection(siamese, training_pairs, use_cuda)
            fine_tuning_data.add_negatives(references)

            logger.info("Beginning fine tuning pass {0}...".format(fine_tuning_pass))
            training_mrrs = np.zeros(n_epochs)
            validation_mrrs = np.zeros(n_epochs)
            models = train_network(siamese, fine_tuning_data, criterion, optimizer, n_epochs, use_cuda)
            for epoch, (model, loss) in enumerate(models):
                utils.utils.save_model(model, model_path.format(fine_tuning_pass, epoch))

                logger.debug("Calculating MRRs...")
                training_mrr, training_mrr_var = utils.experimentation.mean_reciprocal_ranks(model, training_pairs, use_cuda)
                val_mrr, val_mrr_var = utils.experimentation.mean_reciprocal_ranks(model, validation_pairs, use_cuda)
                logger.info("MRRs at pass {0}, epoch {1}:\n\ttrn = {2}\n\tval = {3}".format(fine_tuning_pass, epoch, training_mrr, val_mrr))
                logger.info("Loss at pass {0}, epoch {1} = {2}".format(fine_tuning_pass, epoch, loss.mean()))
                training_mrrs[epoch] = training_mrr
                validation_mrrs[epoch] = val_mrr

            utils.utils.load_model(siamese, model_path.format(fine_tuning_pass, np.argmax(validation_mrrs)))
            utils.utils.save_model(siamese, model_path.format(fine_tuning_pass, 'best'))
            best_validation_mrrs.append(np.max(validation_mrrs))
            best_training_mrrs.append(np.max(training_mrrs))

            utils.graphing.mrr_per_epoch(best_training_mrrs, best_validation_mrrs, title='MRR vs. Epoch (Fine Tuning)')

            fine_tuning_pass += 1

        utils.utils.load_model(siamese, model_path.format(np.argmax(best_validation_mrrs), 'best'))
        utils.utils.save_model(siamese, model_path.format('best', 'best'))

        rrs = utils.experimentation.reciprocal_ranks(siamese, testing_pairs, use_cuda)
        logger.info("Results from best model generated after tine-tuning, evaluated on test data:")
        utils.utils.log_final_stats(rrs)
    except Exception as e:
        utils.utils.save_model(siamese, model_path)
        print("Exception occurred while training: {0}".format(str(e)))
        print(traceback.print_exc())
        exit(1)


def train_network(model, data, objective, optimizer, n_epochs, use_cuda, batch_size=128):
    for epoch in range(n_epochs):
        # if we're using all positives and random negatives, choose new negatives on each epoch
        if isinstance(data, vocal_sketch_data.AllPositivesRandomNegatives):
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
            batch_losses[i] = loss.data[0]

            bar.next()
        bar.finish()

        yield model, batch_losses


def main():
    # set up logger
    logger = logging.getLogger('logger')
    utils.utils.configure_logger(logger)

    # parse arguments
    parser = argparse.ArgumentParser()
    utils.utils.configure_parser(parser)
    args = parser.parse_args()

    logger.info('Beginning experiment...')
    logger.info("CUDA {0}...".format("enabled" if args.cuda else "disabled"))

    try:
        if args.spectrograms:
            utils.preprocessing.load_data_set()
        vocal_sketch = vocal_sketch_files.VocalSketch(*args.partitions)
        train_fine_tuning(args.cuda, vocal_sketch, use_cached_baseline=args.cache_baseline, minimum_passes=args.fine_tuning_passes)
    except Exception as e:
        logger.critical("Unhandled exception: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)


if __name__ == "__main__":
    main()
