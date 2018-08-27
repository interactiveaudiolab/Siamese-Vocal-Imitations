import argparse
import logging
import sys
import traceback

import audaugio

# MUST COME FIRST
# noinspection PyUnresolvedReferences
import utils.matplotlib_backend_hack
import experiments.pairwise
import experiments.triplet
import utils.network
import utils.utils as utilities
from data_files.vocal_imitation import VocalImitation
from data_files.vocal_sketch import VocalSketch_1_1, VocalSketch_1_0
from data_partitions.partitions import Partitions
from data_partitions import PartitionSplit


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
        if cli_args.dataset in ['vs1.0']:
            dataset = VocalSketch_1_0
        elif cli_args.dataset in ['vs1.1']:
            dataset = VocalSketch_1_1
        elif cli_args.dataset in ['vi']:
            dataset = VocalImitation
        else:
            raise ValueError("Invalid dataset ({0}) chosen.".format(cli_args.siamese_dataset))

        imitation_augmentations = audaugio.CombinatoricChain(
            audaugio.BackgroundNoiseAugmentation(.005),
            audaugio.LowPassAugmentation(500, 1.5, 1),
            audaugio.HighPassAugmentation(6000, 1.5, 1),
            audaugio.WindowingAugmentation(4, 2)
        )
        reference_augmentations = audaugio.LinearChain(audaugio.WindowingAugmentation(4, 2))

        datafiles = dataset(recalculate_spectrograms=cli_args.recalculate_spectrograms,
                            imitation_augmentations=imitation_augmentations,
                            reference_augmentations=reference_augmentations)

        data_split = PartitionSplit(*cli_args.partitions)
        partitions = Partitions(datafiles, data_split, cli_args.num_categories, regenerate=cli_args.regenerate_splits or
                                                                                           cli_args.recalculate_spectrograms)
        partitions.save("./output/{0}/partition.pickle".format(utilities.get_trial_number()))

        utils.network.initialize_siamese_params(cli_args.regenerate_weights, cli_args.dropout)

        if cli_args.triplet:
            experiments.triplet.train(cli_args.cuda, cli_args.epochs, cli_args.validation_frequency, cli_args.dropout, partitions, cli_args.optimizer,
                                      cli_args.learning_rate, cli_args.weight_decay, cli_args.momentum)

        if cli_args.pairwise:
            experiments.pairwise.train(cli_args.cuda, cli_args.epochs, cli_args.validation_frequency, cli_args.dropout, partitions, cli_args.optimizer,
                                       cli_args.learning_rate, cli_args.weight_decay, cli_args.momentum)

        cli_args.trials -= 1
        if cli_args.trials > 0:
            main(cli_args)
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
