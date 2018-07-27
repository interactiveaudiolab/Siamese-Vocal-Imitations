import argparse
import logging
import traceback

# MUST COME FIRST
# noinspection PyUnresolvedReferences
import utils.matplotlib_backend_hack
import experiments.fine_tuning
import experiments.random_selection
import utils.utils as utilities
from data_files.vocal_imitation import VocalImitation
from data_files.vocal_sketch import VocalSketch_v2, VocalSketch_v1
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

    # log all CLI args
    logger.debug("\tCLI args:")
    for key in vars(cli_args):
        logger.debug("\t{0} = {1}".format(key, vars(cli_args)[key]))

    try:
        if cli_args.siamese_dataset in ['vs1.0']:
            dataset = VocalSketch_v1
        elif cli_args.siamese_dataset in ['vs2.0']:
            dataset = VocalSketch_v2
        elif cli_args.siamese_dataset in ['vi']:
            dataset = VocalImitation
        else:
            raise ValueError("Invalid dataset ({0}) chosen.".format(cli_args.siamese_dataset))

        siamese_datafiles = dataset(recalculate_spectrograms=cli_args.spectrograms)

        data_split = DataSplit(*cli_args.data_partitions)
        if cli_args.random_only:
            experiments.random_selection.train(cli_args.cuda, siamese_datafiles, cli_args.dropout, cli_args.validate_every, data_split,
                                               cli_args.regenerate_splits, cli_args.regenerate_weights)
        else:
            experiments.fine_tuning.train(cli_args.cuda, siamese_datafiles, cli_args.dropout, cli_args.validate_every, data_split,
                                          cli_args.regenerate_splits, cli_args.regenerate_weights,
                                          use_cached_baseline=cli_args.cache_baseline,
                                          minimum_passes=cli_args.fine_tuning_passes)
        cli_args.trials -= 1
        if cli_args.trials > 0:
            main(cli_args)
    except Exception as e:
        logger.critical("Unhandled exception: {0}".format(str(e)))
        logger.critical(traceback.print_exc())
        exit(1)


if __name__ == "__main__":
    main()
