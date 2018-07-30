import numpy as np

from data_partitions.generics import Partition
from utils.progress_bar import Bar


class TripletPartition(Partition):
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

        self.positive_fine = []
        self.positive_coarse = []
        self.negative_fine = []
        self.negative_coarse = []
        self.all_pairs = []
        shape = [len(self.imitations), len(self.references), len(self.references)]
        self.all_labels = np.zeros(shape)
        n = 0
        update_bar_every = 1000
        bar = Bar("Creating pairs for {0}...".format(dataset_type), max=len(self.references) ** 2 * len(self.imitations))
        for i, (imitation, imitation_label) in enumerate(zip(self.imitations, self.imitation_labels)):
            for j, (near, near_label) in enumerate(zip(self.references, self.reference_labels)):
                for k, (far, far_label) in enumerate(zip(self.references, self.reference_labels)):
                    pair_label = None

                    if near_label['label'] == imitation_label and near_label['is_canonical']:
                        pair_label = 1
                        # near target, far fine
                        if far_label['label'] == imitation_label:
                            self.positive_fine.append([imitation, near, far, pair_label])
                        # near target, far coarse
                        else:
                            self.positive_coarse.append([imitation, near, far, pair_label])
                    elif far_label['label'] == imitation_label and far_label['is_canonical']:
                        pair_label = 0
                        # far target, near fine
                        if near_label['label'] == imitation_label:
                            self.negative_fine.append([imitation, near, far, pair_label])
                        # far target, near coarse
                        else:
                            self.negative_coarse.append([imitation, near, far, pair_label])

                    self.all_pairs.append([imitation, near, far, pair_label])
                    self.all_labels[i, j, k] = pair_label
                    n += 1
                    if n % update_bar_every == 0:
                        bar.next(n=update_bar_every)
        bar.finish()
