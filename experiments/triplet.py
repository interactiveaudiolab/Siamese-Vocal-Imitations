import logging
import traceback

import numpy as np
from torch.nn import BCELoss

from data_partitions.generics import Partitions
from data_partitions.pair import PairPartition
from data_partitions.triplet import TripletPartition
from data_sets.pair import AllPairs
from data_sets.triplet import Balanced
from models.siamese import Siamese
from models.triplet import Triplet
from utils import utils as utilities, training as training, experimentation as experimentation
from utils.obj import TrainingProgress
from utils.utils import initialize_weights, get_optimizer, get_trial_number


def train(use_cuda: bool, n_epochs: int, validate_every: int, use_dropout: bool, partitions: Partitions, optimizer_name: str, lr: float, wd: float,
          momentum: bool):
    logger = logging.getLogger('logger')

    model_path = "./model_output/triplet/model_{0}"
    no_test = True

    partitions.generate_partitions(TripletPartition, no_test=True)
    training_data = Balanced(partitions.train)
    validation_data = Balanced(partitions.val)

    if validate_every > 0:
        partitions.generate_partitions(PairPartition, no_test=no_test)
        training_pairs = AllPairs(partitions.train)
        search_length = training_pairs.n_references
        validation_pairs = AllPairs(partitions.val)
        testing_pairs = AllPairs(partitions.test) if not no_test else None
    else:
        training_pairs = None
        validation_pairs = None
        testing_pairs = None
        search_length = None

    siamese = initialize_weights(Siamese(dropout=use_dropout), use_cuda)
    network = Triplet(dropout=use_dropout)
    network.load_siamese(siamese)

    if use_cuda:
        network = network.cuda()

    criterion = BCELoss()
    optimizer = get_optimizer(network, optimizer_name, lr, wd, momentum)

    try:
        logger.info("Training triplet network...")
        progress = TrainingProgress()
        models = training.train_triplet_network(network, training_data, criterion, optimizer, n_epochs, use_cuda)
        for epoch, (model, training_batch_losses) in enumerate(models):
            utilities.save_model(model, model_path.format(epoch))

            training_loss = training_batch_losses.mean()
            if validate_every != 0 and epoch % validate_every == 0:
                validation_batch_losses = experimentation.triplet_loss(model, validation_data, criterion, use_cuda, batch_size=64)
                validation_loss = validation_batch_losses.mean()

                training_mrr, training_rank = experimentation.mean_reciprocal_ranks(network.siamese, training_pairs, use_cuda)
                val_mrr, val_rank = experimentation.mean_reciprocal_ranks(network.siamese, validation_pairs, use_cuda)

                progress.add_mrr(train=training_mrr, val=val_mrr)
                progress.add_rank(train=training_rank, val=val_rank)
                progress.add_loss(train=training_loss, val=validation_loss)
            else:
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
            logger.info("Results from best model generated during training, evaluated on test data:")
            rrs = experimentation.reciprocal_ranks(network, testing_pairs, use_cuda)
            utilities.log_final_stats(rrs)

        progress.pearson(log=True)
        progress.save("./output/{0}/triplet.pickle".format(get_trial_number()))
        return network
    except Exception as e:
        utilities.save_model(network, model_path.format('crash_backup'))
        logger.critical("Exception occurred while training: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        sys.exit()
