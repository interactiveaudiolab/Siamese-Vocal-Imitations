import logging
import sys
import traceback

import numpy as np
from torch.nn import BCELoss

import utils.network
from data_partitions.partitions import Partitions
from data_partitions.pair_partition import PairPartition
from data_partitions.triplet_partition import TripletPartition
from data_sets.pair import AllPairs
from data_sets.triplet import Balanced
from experiments import Experiment
from models.siamese import Siamese
from models.triplet import Triplet
from utils import utils as utilities, training, inference
from utils.obj import TrainingProgress
from utils.utils import get_optimizer, get_trial_number
from utils.network import initialize_weights


class TripletExperiment(Experiment):
    def __init__(self, use_cuda: bool, n_epochs: int, validate_every: int, use_dropout: bool, partitions: Partitions,
                 optimizer_name: str, lr: float, wd: float, momentum: bool):
        super().__init__(use_cuda, n_epochs, validate_every, use_dropout, partitions, optimizer_name, lr, wd, momentum)

    def train(self):
        logger = logging.getLogger('logger')

        model_path = "./model_output/triplet/model_{0}"
        no_test = True

        self.partitions.generate_partitions(TripletPartition, no_test=True)
        training_data = Balanced(self.partitions.train)
        validation_data = Balanced(self.partitions.val)

        if self.validate_every > 0:
            self.partitions.generate_partitions(PairPartition, no_test=no_test)
            training_pairs = AllPairs(self.partitions.train)
            search_length = training_pairs.n_references
            validation_pairs = AllPairs(self.partitions.val)
            testing_pairs = AllPairs(self.partitions.test) if not no_test else None
        else:
            training_pairs = None
            validation_pairs = None
            testing_pairs = None
            search_length = np.nan

        siamese = initialize_weights(Siamese(dropout=self.use_dropout), self.use_cuda)
        network = Triplet(dropout=self.use_dropout)
        network.load_siamese(siamese)

        if self.use_cuda:
            network = network.cuda()

        criterion = BCELoss()
        optimizer = get_optimizer(network, self.optimizer_name, self.lr, self.wd, self.momentum)

        try:
            logger.info("Training network with triplet loss...")
            progress = TrainingProgress()
            models = training.train_triplet_network(network, training_data, criterion, optimizer, self.n_epochs, self.use_cuda)
            for epoch, (model, training_batch_losses) in enumerate(models):
                utils.network.save_model(model, model_path.format(epoch))

                training_loss = training_batch_losses.mean()
                if self.validate_every != 0 and epoch % self.validate_every == 0:
                    validation_batch_losses = inference.triplet_loss(model, validation_data, criterion, self.use_cuda, batch_size=64)
                    validation_loss = validation_batch_losses.mean()

                    training_mrr, training_rank = inference.mean_reciprocal_ranks(network.siamese, training_pairs, self.use_cuda)
                    val_mrr, val_rank = inference.mean_reciprocal_ranks(network.siamese, validation_pairs, self.use_cuda)

                    progress.add_mrr(train=training_mrr, val=val_mrr)
                    progress.add_rank(train=training_rank, val=val_rank)
                    progress.add_loss(train=training_loss, val=validation_loss)
                else:
                    progress.add_mrr(train=np.nan, val=np.nan)
                    progress.add_rank(train=np.nan, val=np.nan)
                    progress.add_loss(train=training_loss, val=np.nan)

                progress.graph("Triplet", search_length)

            # load weights from best model if we validated throughout
            if self.validate_every > 0:
                network = network.train()
                utils.network.load_model(network, model_path.format(np.argmax(progress.val_mrr)))

            # otherwise just save most recent model
            utils.network.save_model(network, model_path.format('best'))
            utils.network.save_model(network, './output/{0}/triplet'.format(utilities.get_trial_number()))

            if not no_test:
                logger.info("Results from best model generated during training, evaluated on test data:")
                rrs = inference.reciprocal_ranks(network, testing_pairs, self.use_cuda)
                utilities.log_final_stats(rrs)

            progress.pearson(log=True)
            progress.save("./output/{0}/triplet.pickle".format(get_trial_number()))
            return network
        except Exception as e:
            utils.network.save_model(network, model_path.format('crash_backup'))
            logger.critical("Exception occurred while training: {0}".format(str(e)))
            logger.critical(traceback.print_exc())
            sys.exit()
