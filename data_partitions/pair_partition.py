import numpy as np

from data_partitions.generics import Partition
from utils.progress_bar import Bar


class PairPartition(Partition):
    def __init__(self, references, reference_labels, all_imitations, all_imitation_labels, dataset_type):
        super().__init__()
        self.references = references
        self.reference_labels = reference_labels

        reference_label_list = [v['label'] for v in reference_labels]

        # filter out imitations of things that are not in this set at all
        self.imitations = []
        self.imitation_labels = []
        for i, l in zip(all_imitations, all_imitation_labels):
            if l in reference_label_list:
                self.imitations.append(i)
                self.imitation_labels.append(l)

        bar = Bar("Creating pairs for {0}...".format(dataset_type), max=len(self.references) * len(self.imitations))

        self.negative_fine = []
        self.negative_coarse = []
        self.positive = []

        self.all_pairs = []
        self.labels = np.zeros([len(self.imitations), len(self.references)])
        self.canonical_locations = np.zeros([len(self.imitations), len(self.references)])
        n = 0
        update_bar_every = 1000
        for i, (imitation, imitation_label) in enumerate(zip(self.imitations, self.imitation_labels)):
            for j, (reference, reference_label) in enumerate(zip(self.references, self.reference_labels)):
                if reference_label['is_canonical']:
                    self.canonical_locations[i, j] = 1

                if reference_label['label'] == imitation_label:
                    if reference_label['is_canonical']:
                        self.labels[i, j] = 1
                        self.positive.append([imitation, reference, True])
                        self.all_pairs.append([imitation, reference, True])
                    else:
                        self.negative_fine.append([imitation, reference, False])
                        self.all_pairs.append([imitation, reference, False])
                else:
                    self.negative_coarse.append([imitation, reference, False])
                    self.all_pairs.append([imitation, reference, False])

                n += 1
                if n % update_bar_every == 0:
                    bar.next(n=update_bar_every)
        bar.finish()
