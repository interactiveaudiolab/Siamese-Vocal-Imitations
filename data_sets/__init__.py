import itertools
import logging
import os
from typing import Union

import numpy as np

from exceptions import NoSpectrogramOnDiskError
from utils.utils import get_npy_dir


class Dataset:
    def __init__(self, name):
        self.name = name

        self.references = None
        self.reference_labels = None
        self.imitations = None
        self.imitation_labels = None

        self.load_from_disk()

    def load_from_disk(self):
        logger = logging.getLogger('logger')
        try:
            for i in itertools.count():
                logger.debug("Loading batch {0}...".format(i))
                self.references = self.add_npy(self.references, self.load_npy("references_{0}.npy".format(i)))
                self.imitations = self.add_npy(self.imitations, self.load_npy("imitations_{0}.npy".format(i)))
                self.reference_labels = self.add_npy(self.reference_labels, self.load_npy("references_{0}_labels.npy".format(i)))
                self.imitation_labels = self.add_npy(self.imitation_labels, self.load_npy("imitations_{0}_labels.npy".format(i)))
        except FileNotFoundError:
            # noinspection PyUnboundLocalVariable
            if i == 0:  # nothing found
                raise NoSpectrogramOnDiskError
            else:
                logger.info("Found {0} batches of .npy arrays".format(i))

    def load_npy(self, file_name):
        path = os.path.join(get_npy_dir(self.name), file_name)
        return np.load(path)

    @staticmethod
    def add_npy(ar: Union[None, np.ndarray], to_add):
        if ar is None:
            return to_add
        else:
            return np.concatenate((ar, to_add))
