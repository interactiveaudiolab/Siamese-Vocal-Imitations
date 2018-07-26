import logging

from utils import utils


class Datafiles:
    def __init__(self, version, shuffle, recalculate_spectrograms):
        logger = logging.getLogger('logger')
        logger.info("Using dataset: {0}".format(version))

        if recalculate_spectrograms:
            logger.info("Calculating spectrograms for {0}".format(version))
            self.calculate_spectrograms()

        try:
            self.references = utils.load_npy("references.npy", version)
            self.reference_labels = utils.load_npy("references_labels.npy", version)

            self.imitations = utils.load_npy("imitations.npy", version)
            self.imitation_labels = utils.load_npy("imitations_labels.npy", version)
        except FileNotFoundError:
            logger.warning("No saved spectrograms for {0}. Calculating them...".format(version))
            self.calculate_spectrograms()

            self.references = utils.load_npy("references.npy", version)
            self.reference_labels = utils.load_npy("references_labels.npy", version)

            self.imitations = utils.load_npy("imitations.npy", version)
            self.imitation_labels = utils.load_npy("imitations_labels.npy", version)

        if shuffle:
            self.references, self.reference_labels = utils.zip_shuffle(self.references, self.reference_labels)
            self.imitations, self.imitation_labels = utils.zip_shuffle(self.imitations, self.imitation_labels)

    @staticmethod
    def calculate_spectrograms():
        raise NotImplementedError
