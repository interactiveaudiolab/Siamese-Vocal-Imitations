import logging
import os
from typing import Tuple, Dict, List

import numpy as np

from utils import preprocessing as preprocessing
from utils.utils import get_npy_dir


class Datafiles:
    def __init__(self, name, imitation_augmentations, reference_augmentations, recalculate_spectrograms=False):
        self.imitation_augmentations = imitation_augmentations
        self.reference_augmentations = reference_augmentations
        self.name = name
        self.references = None
        self.reference_labels = None
        self.imitations = None
        self.imitation_labels = None

        logger = logging.getLogger('logger')
        logger.info("Using dataset: {0}".format(name))

        if recalculate_spectrograms:
            logger.info("Calculating spectrograms for {0}...".format(name))
            self.calculate_spectrograms()

        try:
            self.load_from_disk()
        except FileNotFoundError:
            logger.warning("No saved spectrograms found for {0}. Calculating them...".format(name))
            self.calculate_spectrograms()
            self.load_from_disk()

    def load_from_disk(self):
        self.references = self.load_npy("references.npy", self.name)
        self.reference_labels = self.load_npy("references_labels.npy", self.name)
        self.imitations = self.load_npy("imitations.npy", self.name)
        self.imitation_labels = self.load_npy("imitations_labels.npy", self.name)

    def prepare_spectrogram_calculation(self) -> Tuple[Dict, List, Dict, List]:
        """
        Prepare lists of paths to audio files that are to have spectrograms calculated, as well as dictionaries of their labels (indexed by path).
        """
        raise NotImplementedError

    def calculate_spectrograms(self):
        imitation_labels, imitation_paths, reference_labels, reference_paths = self.prepare_spectrogram_calculation()

        preprocessing.calculate_spectrograms(imitation_paths,
                                             imitation_labels,
                                             'imitations',
                                             self.name,
                                             preprocessing.imitation_spectrogram,
                                             self.imitation_augmentations)

        preprocessing.calculate_spectrograms(reference_paths,
                                             reference_labels,
                                             'references',
                                             self.name,
                                             preprocessing.reference_spectrogram,
                                             self.imitation_augmentations)

    @staticmethod
    def load_npy(file_name, dataset):
        path = os.path.join(get_npy_dir(dataset), file_name)
        return np.load(path)
