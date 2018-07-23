import argparse
import logging
import traceback

# MUST COME FIRST
# noinspection PyUnresolvedReferences
import utils.matplotlib_backend_hack
import experiments.fine_tuning
import experiments.random_selection
import experiments.transfer_learning
import utils.utils as utilities

from datafiles.voxforge import Voxforge
from datafiles.urban_sound_8k import UrbanSound8K
from datafiles.vocal_sketch import VocalSketch_v2


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
    for key in vars(cli_args):
        logger.debug("\t{0} = {1}".format(key, vars(cli_args)[key]))

    try:
        vocal_sketch = VocalSketch_v2(*cli_args.partitions, recalculate_spectrograms=cli_args.spectrograms)
        if cli_args.random_only:
            experiments.random_selection.train(cli_args.cuda, vocal_sketch, cli_args.dropout, cli_args.no_normalization)
        else:
            experiments.fine_tuning.train(cli_args.cuda, vocal_sketch, cli_args.dropout, cli_args.no_normalization, use_cached_baseline=cli_args.cache_baseline,
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
