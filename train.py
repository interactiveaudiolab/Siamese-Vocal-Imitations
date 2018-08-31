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
from data_partitions import PartitionSplit, Partitions
from preprocessing import Preprocessor


def main(cli_args=None):
    utilities.update_trial_number()
    utilities.create_output_directory()

    logger = logging.getLogger('logger')
    file_only_logger = logging.getLogger('file.logger')
    parser = argparse.ArgumentParser()
    utilities.configure_parser(parser)
    utilities.configure_logger(logger)
    utilities.configure_logger(file_only_logger, console=False)
    if cli_args is None:
        cli_args = parser.parse_args()

    logger.info('Beginning trial #{0}...'.format(utilities.get_trial_number()))
    log_cli_args(cli_args)
    try:
        # un-processed audio files on disk
        datafiles = get_datafiles(cli_args.dataset)

        # spectrogram recalculation is manually specified with a CLI arg
        if cli_args.recalculate_spectrograms:
            imitation_augmentations, reference_augmentations = get_augmentation_chains()
            preprocessor = Preprocessor(imitation_augmentations, reference_augmentations)
            preprocessor(datafiles)

        # processed spectrograms
        data_split = PartitionSplit(*cli_args.partitions, cli_args.num_categories)
        partitions = Partitions(datafiles.name, data_split)
        partitions.save(utilities.get_trial_directory("partition.pickle"))

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


def get_datafiles(dataset_name):
    if dataset_name in ['vs1.0']:
        datafiles = VocalSketch_1_0()
    elif dataset_name in ['vs1.1']:
        datafiles = VocalSketch_1_1()
    elif dataset_name in ['vi']:
        datafiles = VocalImitation()
    else:
        raise ValueError("Invalid datafiles ({0}) chosen.".format(dataset_name))
    return datafiles


def get_augmentation_chains():
    windowing_augmentation = audaugio.WindowingAugmentation(4, 2)
    imitation_augmentations = audaugio.CombinatoricChain(
        audaugio.BackgroundNoiseAugmentation(.005),
        audaugio.LowPassAugmentation(500, 1.5, 1),
        audaugio.HighPassAugmentation(6000, 1.5, 1),
        windowing_augmentation
    )
    reference_augmentations = audaugio.LinearChain(windowing_augmentation)
    return imitation_augmentations, reference_augmentations


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
