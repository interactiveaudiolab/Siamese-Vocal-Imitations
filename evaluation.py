import argparse
import logging
import sys
import traceback

# MUST COME FIRST
# noinspection PyUnresolvedReferences
import numpy as np

import utils.matplotlib_backend_hack
import utils.network
import utils.utils as utilities
from data_files.vocal_imitation import VocalImitation
from data_partitions.generics import DataSplit
from data_partitions.pair_partition import PairPartition
from data_partitions.partitions import Partitions
from data_sets.pair import AllPairs
from models.siamese import Siamese
from models.triplet import Triplet
from utils.inference import canonical_mean_recall


def main(cli_args=None):
    utilities.update_trial_number()
    utilities.create_output_directory()

    logger = logging.getLogger('logger')
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    utilities.configure_parser(parser)
    utilities.configure_logger(logger)
    if cli_args is None:
        cli_args = parser.parse_args()

    logger.info('Beginning trial #{0}...'.format(utilities.get_trial_number()))
    log_cli_args(cli_args)
    try:
        datafiles = VocalImitation(recalculate_spectrograms=cli_args.recalculate_spectrograms)
        data_split = DataSplit(*cli_args.partitions)
        partitions = Partitions(datafiles, data_split, cli_args.num_categories, regenerate_splits=False)
        partitions.generate_partitions(PairPartition, no_test=True)
        partitions.save("./output/{0}/partition.pickle".format(utilities.get_trial_number()))

        if cli_args.triplet:
            model = Triplet(dropout=cli_args.dropout)
        elif cli_args.pairwise:
            model = Siamese(dropout=cli_args.dropout)
        else:
            raise ValueError("You must specify the type of the model that is to be evaluated (triplet or pairwise")

        # show_model(model)

        if cli_args.cuda:
            model = model.cuda()

        utils.network.load_model(model, cli_args.model_path, cli_args.cuda)
        recall = canonical_mean_recall(model if cli_args.pairwise else model.siamese, AllPairs(partitions.train), cli_args.cuda, cli_args.num_categories)
        logger.info("recall = {0}".format(recall))
    except Exception as e:
        logger.critical("Unhandled exception: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        sys.exit()


def show_model(model):
    print(model)
    num_parameters = 0
    for p in model.parameters():
        if p.requires_grad:
            num_parameters += np.cumprod(p.size())[-1]
    print('Number of parameters: %d' % num_parameters)


def log_cli_args(cli_args):
    logger = logging.getLogger('logger')
    logger.debug("\tCLI args:")
    cli_arg_dict = vars(cli_args)
    keys = list(cli_arg_dict.keys())
    keys.sort()
    for key in keys:
        logger.debug("\t{0} = {1}".format(key, cli_arg_dict[key]))


if __name__ == "__main__":
    main()
