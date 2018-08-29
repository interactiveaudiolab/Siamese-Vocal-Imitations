import logging
from typing import Tuple, Dict, List


class Datafiles:
    def __init__(self, name):
        logger = logging.getLogger('logger')
        logger.info("Using dataset: {0}".format(name))

        self.name = name

        self.imitation_path_labels = {}
        self.imitation_paths = []
        self.reference_path_labels = {}
        self.reference_paths = []

    def prepare_spectrogram_calculation(self) -> Tuple[Dict, List, Dict, List]:
        """
        Prepare lists of paths to audio files that are to have spectrograms calculated, as well as dictionaries of their labels (indexed by path).
        """
        raise NotImplementedError
