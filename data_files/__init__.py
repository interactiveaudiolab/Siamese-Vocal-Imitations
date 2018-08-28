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
        self.n_batches = 50

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
        for i in range(self.n_batches):
            self.references += self.load_npy("references_{0}.npy".format(i), self.name)
            self.imitations += self.load_npy("imitations_{0}.npy".format(i), self.name)
        self.reference_labels = self.load_npy("references_labels.npy", self.name)
        self.imitation_labels = self.load_npy("imitations_labels.npy", self.name)

    def prepare_spectrogram_calculation(self) -> Tuple[Dict, List, Dict, List]:
        """
        Prepare lists of paths to audio files that are to have spectrograms calculated, as well as dictionaries of their labels (indexed by path).
        """
        raise NotImplementedError

    def calculate_spectrograms(self):
        imitation_labels, imitation_paths, reference_labels, reference_paths = self.prepare_spectrogram_calculation()

        logger = logging.getLogger('logger')
        logger.info("Calculating spectrograms in {0} batches".format(self.n_batches))

        batches = np.array_split(imitation_paths, self.n_batches)
        for i, batch in enumerate(batches):
            preprocessing.calculate_spectrograms(batch,
                                                 imitation_labels,
                                                 'imitations_{0}'.format(i),
                                                 self.name,
                                                 preprocessing.imitation_spectrogram,
                                                 self.imitation_augmentations)

        n_batches = int(len(reference_paths) / self.n_batches)
        batches = np.array_split(reference_paths, n_batches)
        for i, batch in enumerate(batches):
            preprocessing.calculate_spectrograms(batch,
                                                 reference_labels,
                                                 'references_{0}'.format(i),
                                                 self.name,
                                                 preprocessing.reference_spectrogram,
                                                 self.imitation_augmentations)

    @staticmethod
    def load_npy(file_name, dataset):
        path = os.path.join(get_npy_dir(dataset), file_name)
        return np.load(path)
