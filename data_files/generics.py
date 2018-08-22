import logging

from utils import utils


class Datafiles:
    def __init__(self, name, augmentations=None, recalculate_spectrograms=False):
        self.augmentations = augmentations
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
        self.references = utils.load_npy("references.npy", self.name)
        self.reference_labels = utils.load_npy("references_labels.npy", self.name)
        self.imitations = utils.load_npy("imitations.npy", self.name)
        self.imitation_labels = utils.load_npy("imitations_labels.npy", self.name)

    def calculate_spectrograms(self):
        """
        Calculates normalized imitation and reference spectrograms and saves them as .npy files.
        """
        raise NotImplementedError
