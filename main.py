import argparse
import logging
import sys
import traceback

# MUST COME FIRST
# noinspection PyUnresolvedReferences
import utils.matplotlib_backend_hack
import experiments.siamese
import experiments.triplet
import utils.utils as utilities
from data_files.vocal_imitation import VocalImitation
from data_files.vocal_sketch import VocalSketchV2, VocalSketchV1
from data_partitions.generics import Partitions
from utils.obj import DataSplit


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
            dataset = VocalSketchV1
        elif cli_args.dataset in ['vs2.0']:
            dataset = VocalSketchV2
        elif cli_args.dataset in ['vi']:
            dataset = VocalImitation
        else:
            raise ValueError("Invalid dataset ({0}) chosen.".format(cli_args.siamese_dataset))

        datafiles = dataset(recalculate_spectrograms=cli_args.recalculate_spectrograms)
        data_split = DataSplit(*cli_args.partitions)
        partitions = Partitions(datafiles, data_split, cli_args.num_categories, regenerate_splits=cli_args.regenerate_splits)
        if cli_args.regenerate_weights:
            utilities.regenerate_siamese_weights(cli_args.dropout)

        if cli_args.triplet:
            experiments.triplet.train(cli_args.cuda, cli_args.epochs, cli_args.validation_frequency, cli_args.dropout, partitions, cli_args.optimizer,
                                      cli_args.learning_rate, cli_args.weight_decay, cli_args.momentum)

        if cli_args.siamese:
            experiments.siamese.train(cli_args.cuda, cli_args.epochs, cli_args.validation_frequency, cli_args.dropout, partitions, cli_args.optimizer,
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
