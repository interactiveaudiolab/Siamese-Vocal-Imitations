import logging
import os
import pickle
import sys


class Partition:
    def __init__(self, references, reference_labels, all_imitations, all_imitation_labels):
        pass


class PartitionSplit:
    def __init__(self, train_ratio: float, validation_ratio: float, test_ratio: float = None):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        if test_ratio:
            self.test_ratio = test_ratio
        else:
            self.test_ratio = 1 - (train_ratio + validation_ratio)


class PartitionArguments:
    def __init__(self, references, reference_labels, imitations, imitation_labels, partition_name):
        self.references = references
        self.reference_labels = reference_labels
        self.imitations = imitations
        self.imitation_labels = imitation_labels
        self.partition_name = partition_name


class SaveablePartitionState:
    def __init__(self):
        self.shuffled_category_indices = None
        self.shuffled_imitation_indices = None
        self.split = None
        self.n_train_val_categories = None
        self.categories = None

    def load(self, location):
        logger = logging.getLogger('logger')

        try:
            logger.debug("Loading partitions from {0}...".format(location))
            with open(location, 'rb') as f:
                self.shuffled_imitation_indices = pickle.load(f)
                self.shuffled_category_indices = pickle.load(f)

        except FileNotFoundError:
            with open(location, 'w+b'):
                logger.critical("No pickled partition at {0}".format(location))
                sys.exit()
        except EOFError:
            logger.critical("Insufficient amount of data found in {0}".format(location))
            sys.exit()

    def save(self, location):
        logger = logging.getLogger('logger')

        logger.debug("Saving partitions at {0}...".format(location))
        with open(location, 'wb') as f:
            pickle.dump(self.shuffled_imitation_indices, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.shuffled_category_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

        if not self.n_train_val_categories:
            n_train_val = int(self.split.train_ratio * len(self.categories) + self.split.validation_ratio * len(self.categories))
        else:
            n_train_val = self.n_train_val_categories

        categories = self.categories[self.shuffled_category_indices][:n_train_val]

        category_list_path = os.path.join(os.path.dirname(location), 'categories.txt')
        with open(category_list_path, 'w+') as f:
            f.write("\n".join(categories))