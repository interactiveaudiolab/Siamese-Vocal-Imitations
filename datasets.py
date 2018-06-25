import numpy as np
import torch.utils.data.dataset as dataset
from progress.bar import Bar

from preprocessing import normalize_spectrograms
from utils import load_npy


class FineTuned(dataset.Dataset):
    def __init__(self):
        self.all_positives = []

        references = load_npy("references.npy")
        self.references = normalize_spectrograms(references)
        reference_labels = load_npy("reference_labels.npy")

        imitations = load_npy("imitations.npy")
        self.imitations = normalize_spectrograms(imitations)
        imitation_labels = load_npy("imitation_labels.npy")

        bar = Bar("Creating training pairs", max=len(references) * len(imitations))
        for reference, reference_label in zip(references, reference_labels):
            for imitation, imitation_label in zip(imitations, imitation_labels):
                if reference_label == imitation_label:
                    self.all_positives.append([imitation, reference])
                bar.next()
        bar.finish()

        self.pairs = []

    def add_negatives(self, reference_indexes):
        for i, r in enumerate(reference_indexes):
            self.add_negative(self.imitations[i], self.references[r])

    def add_negative(self, i, r):
        self.pairs.append([i, r, False])

    def reset(self):
        self.pairs = []
        for i, r in self.all_positives:
            self.pairs.append([i, r, True])

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


class AllPositivesRandomNegatives(dataset.Dataset):
    def __init__(self, limit):
        positives, negatives = self.get_all_training_pairs(limit)
        self.positives = positives
        self.negatives = negatives
        self.pairs = []
        self.reselect_negatives()

    def reselect_negatives(self):
        # clear out selected negatives
        self.pairs = []
        indices = np.random.choice(np.arange(len(self.negatives)), len(self.positives))
        for i in indices:
            imitation, reference = self.negatives[i]
            self.pairs.append([imitation, reference, False])

        for i, r in self.positives:
            self.pairs.append([i, r, True])

        np.random.shuffle(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def get_all_training_pairs(limit):
        """
        Create all possible pairs of imitation and reference.

        :return: positive_pairs, negative_pairs. Each are arrays of tuples of imitations and reference.
        """
        references = load_npy("references.npy")
        references = normalize_spectrograms(references)
        reference_labels = load_npy("reference_labels.npy")

        imitations = load_npy("imitations.npy")
        imitations = normalize_spectrograms(imitations)
        imitation_labels = load_npy("imitation_labels.npy")

        if limit:
            references = references[:limit]
            imitations = imitations[:limit]

        bar = Bar("Creating all training pairs...", max=len(references) * len(imitations))
        positive_pairs = []  # (imitation spectrogram, reference spectrogram)
        negative_pairs = []
        for reference, reference_label in zip(references, reference_labels):
            for imitation, imitation_label in zip(imitations, imitation_labels):
                if reference_label == imitation_label:
                    positive_pairs.append([imitation, reference])
                else:
                    negative_pairs.append([imitation, reference])

                bar.next()
        bar.finish()

        return positive_pairs, negative_pairs


class AllPairs(dataset.Dataset):
    def __init__(self, limit):
        imitations = load_npy("imitations.npy")
        imitations = normalize_spectrograms(imitations)
        imitation_labels = load_npy("imitation_labels.npy")

        references = load_npy("references.npy")
        references = normalize_spectrograms(references)
        reference_labels = load_npy("reference_labels.npy")

        if limit:
            references = references[:limit]
            imitations = imitations[:limit]

        self.imitations = imitations
        self.references = references

        self.n_imitations = len(imitations)
        self.n_references = len(references)

        bar = Bar("Creating all possible pairs...", max=len(references) * len(imitations))
        self.pairs = []
        self.labels = np.zeros([self.n_imitations, self.n_references])
        for i, (imitation, imitation_label) in enumerate(zip(imitations, imitation_labels)):
            for j, (reference, reference_label) in enumerate(zip(references, reference_labels)):
                if reference_label == imitation_label:
                    self.pairs.append([imitation, reference, True])
                    self.labels[i, j] = 1
                else:
                    self.pairs.append([imitation, reference, False])
                    self.labels[i, j] = 0

                bar.next()
        bar.finish()

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)
