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
from data_files.vocal_sketch import VocalSketchV2
from data_partitions.generics import DataSplit
from data_partitions.pair_partition import PairPartition
from data_partitions.partitions import Partitions
from data_sets.pair import AllPairs
from models.siamese import Siamese
from models.triplet import Triplet
from utils.inference import mean_reciprocal_ranks


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
        dataset = VocalSketchV2
        datafiles = dataset(recalculate_spectrograms=cli_args.recalculate_spectrograms)
        data_split = DataSplit(*[0.01, 0.01, .98])
        partitions = Partitions(datafiles, data_split, cli_args.num_categories, regenerate_splits=cli_args.regenerate_splits or
                                                                                                  cli_args.recalculate_spectrograms)
        partitions.generate_partitions(PairPartition)
        partitions.save("./output/{0}/partition.pickle".format(utilities.get_trial_number()))
        triplet = Triplet(dropout=cli_args.dropout)
        show_model(triplet)
        if cli_args.cuda:
            triplet = triplet.cuda()
        utils.network.load_model(triplet, cli_args.model_path, cli_args.cuda)
        mrr, rank = mean_reciprocal_ranks(triplet.siamese, AllPairs(partitions.test), cli_args.cuda)
        logger.info("MRR = {0}".format(mrr))
        logger.info("rank = {0}".format(rank))
        cli_args.trials -= 1
        if cli_args.trials > 0:
            main(cli_args)
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
