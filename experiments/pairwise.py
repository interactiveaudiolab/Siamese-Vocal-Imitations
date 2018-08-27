import logging
import sys
import traceback

import numpy as np
from torch.nn import BCELoss

import utils.network
from data_partitions.partitions import Partitions
from data_partitions.pair_partition import PairPartition
from data_sets.pair import Balanced, AllPairs
from models.siamese import Siamese
from utils import utils as utilities, training, inference
from utils.obj import TrainingProgress
from utils.utils import get_optimizer
from utils.network import initialize_weights


def train(use_cuda: bool, n_epochs: int, validate_every: int, use_dropout: bool, partitions: Partitions, optimizer_name: str, lr: float, wd: float,
          momentum: bool):
    logger = logging.getLogger('logger')

    no_test = True
    model_path = "./model_output/pairwise/model_{0}"

    partitions.generate_partitions(PairPartition, no_test=no_test)
    training_data = Balanced(partitions.train)

    if validate_every > 0:
        balanced_validation = Balanced(partitions.val)
        training_pairs = AllPairs(partitions.train)
        search_length = training_pairs.n_references
        validation_pairs = AllPairs(partitions.val)
        testing_pairs = AllPairs(partitions.test) if not no_test else None
    else:
        balanced_validation = None
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
        logger.info("Training network with pairwise loss...")
        progress = TrainingProgress()
        models = training.train_siamese_network(siamese, training_data, criterion, optimizer, n_epochs, use_cuda)
        for epoch, (model, training_batch_losses) in enumerate(models):
            utils.network.save_model(model, model_path.format(epoch))

            training_loss = training_batch_losses.mean()
            if validate_every != 0 and epoch % validate_every == 0:
                validation_batch_losses = inference.siamese_loss(model, balanced_validation, criterion, use_cuda)
                validation_loss = validation_batch_losses.mean()

                training_mrr, training_rank = inference.mean_reciprocal_ranks(model, training_pairs, use_cuda)
                val_mrr, val_rank = inference.mean_reciprocal_ranks(model, validation_pairs, use_cuda)

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
            utils.network.load_model(siamese, model_path.format(np.argmax(progress.val_mrr)))

        # otherwise just save most recent model
        utils.network.save_model(siamese, model_path.format('best'))
        utils.network.save_model(siamese, './output/{0}/pairwise'.format(utilities.get_trial_number()))

        if not no_test:
            logger.info("Results from best model generated during training, evaluated on test data:")
            rrs = inference.reciprocal_ranks(siamese, testing_pairs, use_cuda)
            utilities.log_final_stats(rrs)

        progress.pearson(log=True)
        progress.save("./output/{0}/pairwise.pickle".format(utilities.get_trial_number()))
        return siamese
    except Exception as e:
        utils.network.save_model(siamese, model_path.format('crash_backup'))
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        sys.exit()
