import logging
import traceback

import numpy as np
from torch.nn import BCELoss

from data_partitions.generics import Partitions
from data_partitions.pair import PairPartition
from data_sets.pair import Balanced, AllPairs
from models.siamese import Siamese
from utils import utils as utilities, training, experimentation
from utils.obj import TrainingProgress
from utils.utils import initialize_weights, get_optimizer


def train(use_cuda: bool, n_epochs: int, validate_every: int, use_dropout: bool, partitions: Partitions, optimizer_name: str, lr: float, wd: float,
          momentum: bool):
    logger = logging.getLogger('logger')

    no_test = True
    model_path = "./model_output/siamese/model_{0}"

    partitions.generate_partitions(PairPartition, no_test=no_test)
    training_data = Balanced(partitions.train)
    if validate_every > 0:
        training_pairs = AllPairs(partitions.train)
        search_length = training_pairs.n_references
        validation_pairs = AllPairs(partitions.val)
        testing_pairs = AllPairs(partitions.test) if not no_test else None
    else:
        training_pairs = None
        validation_pairs = None
        testing_pairs = None
        search_length = None

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese(dropout=use_dropout)
    siamese = initialize_weights(siamese, use_cuda)

    if use_cuda:
        siamese = siamese.cuda()

    criterion = BCELoss()
    optimizer = get_optimizer(siamese, optimizer_name, lr, wd, momentum)

    try:
        logger.info("Training siamese network...")
        progress = TrainingProgress()
        models = training.train_siamese_network(siamese, training_data, criterion, optimizer, n_epochs, use_cuda)
        for epoch, (model, training_batch_losses) in enumerate(models):
            utilities.save_model(model, model_path.format(epoch))

            training_loss = training_batch_losses.mean()
            if validate_every != 0 and epoch % validate_every == 0:
                validation_batch_losses = experimentation.siamese_loss(model, validation_pairs, criterion, use_cuda)
                validation_loss = validation_batch_losses.mean()

                training_mrr, training_rank = experimentation.mean_reciprocal_ranks(model, training_pairs, use_cuda)
                val_mrr, val_rank = experimentation.mean_reciprocal_ranks(model, validation_pairs, use_cuda)

                progress.add_mrr(train=training_mrr, val=val_mrr)
                progress.add_rank(train=training_rank, val=val_rank)
                progress.add_loss(train=training_loss, val=validation_loss)
            else:
                progress.add_mrr(train=np.nan, val=np.nan)
                progress.add_rank(train=np.nan, val=np.nan)
                progress.add_loss(train=training_loss, val=np.nan)

            progress.graph("Siamese", search_length)
        # load weights from best model if we validated throughout
        if validate_every > 0:
            siamese = siamese.train()
            utilities.load_model(siamese, model_path.format(np.argmax(progress.val_mrr)))
        # otherwise just save most recent model
        utilities.save_model(siamese, model_path.format('best'))
        utilities.save_model(siamese, './output/{0}/siamese'.format(utilities.get_trial_number()))

        if not no_test:
            logger.info("Results from best model generated during random-selection training, evaluated on test data:")
            rrs = experimentation.reciprocal_ranks(siamese, testing_pairs, use_cuda)
            utilities.log_final_stats(rrs)

        train_correlation, val_correlation = progress.pearson()
        logger.info("Correlations between loss and MRR:\n\ttrn = {0}\n\tval = {1}".format(train_correlation, val_correlation))
        progress.save("./output/{0}/siamese.pickle".format(utilities.get_trial_number()))
        return siamese
    except Exception as e:
        utilities.save_model(siamese, model_path.format('crash_backup'))
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)
