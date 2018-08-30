import argparse
import logging
import os
import sys
import traceback

import numpy as np

# MUST COME FIRST
# noinspection PyUnresolvedReferences
import utils.matplotlib_backend_hack
import utils.network
import utils.utils as utilities
from data_files.vocal_imitation import VocalImitation
from data_partitions import PartitionSplit
from data_partitions.pair_partition import PairPartition
from data_partitions.partitions import Partitions
from data_sets import Dataset
from data_subsets.pair import AllPairs
from models.siamese import Siamese
from models.triplet import Triplet
from utils.graphing import num_canonical_memorized
from utils.inference import num_memorized_canonicals


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
    log_cli_args(cli_args)
    try:
        datafiles = VocalImitation()
        dataset = Dataset(datafiles.name)
        data_split = PartitionSplit(*cli_args.partitions)
        partitions = Partitions(dataset, data_split, cli_args.num_categories, regenerate=False)
        partitions.generate_partitions(PairPartition, no_test=True)
        partitions.save(utilities.get_trial_directory("partition.pickle"))

        if cli_args.triplet:
            model = Triplet(dropout=cli_args.dropout)
        elif cli_args.pairwise:
            model = Siamese(dropout=cli_args.dropout)
        else:
            raise ValueError("You must specify the type of the model that is to be evaluated (triplet or pairwise")

        if cli_args.cuda:
            model = model.cuda()

        evaluated_epochs = np.arange(0, 300, step=5)
        model_directory = './output/models//{0}'.format('pairwise' if cli_args.pairwise else 'triplet') + '/model_{0}'
        model_paths = [model_directory.format(n) for n in evaluated_epochs]
        n_memorized = []
        memorized_var = []
        for model_path in model_paths:
            utils.network.load_model(model, model_path, cli_args.cuda)
            n, v = num_memorized_canonicals(model if cli_args.pairwise else model.siamese, AllPairs(partitions.train),
                                            cli_args.cuda)
            logger.info("n = {0}\nv={1}".format(n, v))
            n_memorized.append(n)
            memorized_var.append(v)

            num_canonical_memorized(memorized_var, n_memorized, evaluated_epochs[:len(n_memorized)], cli_args.num_categories)

    except Exception as e:
        logger.critical("Unhandled exception: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        sys.exit()


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
