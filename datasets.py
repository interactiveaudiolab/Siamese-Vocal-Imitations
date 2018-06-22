import numpy as np
import torch.utils.data.dataset as dataset
from progress.bar import Bar

from preprocessing import normalize_spectrograms
from data_utils import load_npy


class FineTuned(dataset.Dataset):
    def __init__(self):
        self.all_positives = self.get_positive_pairs()
        self.pairs = []

        self.reset()

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

    @staticmethod
    def get_positive_pairs():
        references = load_npy("references.npy")
        references = normalize_spectrograms(references)
        reference_labels = load_npy("reference_labels.npy")

        imitations = load_npy("imitations.npy")
        imitations = normalize_spectrograms(imitations)
        imitation_labels = load_npy("imitation_labels.npy")

        bar = Bar("Creating training pairs", max=len(references) * len(imitations))
        pairs = []  # (imitation spectrogram, reference spectrogram)
        for reference, reference_label in zip(references, reference_labels):
            for imitation, imitation_label in zip(imitations, imitation_labels):
                if reference_label == imitation_label:
                    pairs.append([imitation, reference])
                bar.next()
        bar.finish()

        return pairs


class AllPositivesRandomNegatives(dataset.Dataset):
    def __init__(self):
        positives, negatives = self.get_all_training_pairs()
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
    def get_all_training_pairs():
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

        bar = Bar("Creating all training pairs...", max=len(references) * len(imitations))
        positive_pairs = []  # (imitation spectrogram, reference spectrogram)
        negative_pairs = []
        # i = 0
        for reference, reference_label in zip(references, reference_labels):
            for imitation, imitation_label in zip(imitations, imitation_labels):
                if reference_label == imitation_label:
                    positive_pairs.append([imitation, reference])
                    # i += 1
                else:
                    negative_pairs.append([imitation, reference])

                # if i > 10:
                #     return positive_pairs, negative_pairs
                bar.next()
        bar.finish()

        return positive_pairs, negative_pairs


class AllImitations(dataset.Dataset):
    def __init__(self):
        imitations = load_npy("imitations.npy")
        imitations = normalize_spectrograms(imitations)
        self.imitations = imitations

    def __getitem__(self, index):
        return self.imitations[index]

    def __len__(self):
        return len(self.imitations)


class AllReferences(dataset.Dataset):
    def __init__(self):
        references = load_npy("references.npy")
        references = normalize_spectrograms(references)
        self.references = references

    def __getitem__(self, index):
        return self.references[index]

    def __len__(self):
        return len(self.references)


class AllPositivePairs(dataset.Dataset):
    def __init__(self):
        self.all_positives = self.get_positive_pairs()
        self.pairs = []
        for i, r in self.all_positives:
            self.pairs.append([i, r])

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def get_positive_pairs():
        references = load_npy("references.npy")
        references = normalize_spectrograms(references)
        reference_labels = load_npy("reference_labels.npy")

        imitations = load_npy("imitations.npy")
        imitations = normalize_spectrograms(imitations)
        imitation_labels = load_npy("imitation_labels.npy")

        bar = Bar("Creating positive training pairs...", max=len(references) * len(imitations))
        pairs = []  # (imitation spectrogram, reference spectrogram)
        for reference, reference_label in zip(references, reference_labels):
            for imitation, imitation_label in zip(imitations, imitation_labels):
                if reference_label == imitation_label:
                    pairs.append([imitation, reference])
                bar.next()
        bar.finish()

        return pairs


class OneImitationAllReferences(dataset.Dataset):
    def __init__(self):
        """
        Offers pairs of a given imitation and all possible reference recordings. Reference recordings are loaded once per instantiation, so the imitation can
        be updated at low cost.
        """
        references = load_npy("references.npy")
        references = normalize_spectrograms(references)
        self.references = references
        self.imitation = None

    def set_imitation(self, imitation):
        """
        Set the imitation that will be included in all pairs generated by this iterator.

        :param imitation:
        """
        self.imitation = imitation

    def __getitem__(self, index):
        if self.imitation is not None:
            return [self.imitation, self.references[index]]
        else:
            raise RuntimeError("Imitation has not been set. Use .set_imitation before trying to use this iterator.")

    def __len__(self):
        return len(self.references)
