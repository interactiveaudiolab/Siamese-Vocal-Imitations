import logging


class NoSpectrogramOnDiskError(Exception):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        logger = logging.getLogger('logger')
        logger.critical("Missing .npy data for {0}. The spectrograms might need to be (re-)calculated.".format(self.dataset_name))
