import csv
import os

from data_files.generics import Datafiles
from utils import preprocessing
from utils.utils import get_dataset_dir


class VocalImitation(Datafiles):
    def __init__(self, recalculate_spectrograms=False):
        super().__init__('vocal_imitation', recalculate_spectrograms)

    @staticmethod
    def calculate_spectrograms():
        data_dir = get_dataset_dir()
        imitation_path = os.path.join(data_dir, "vocal_imitation/imitations")
        reference_path = os.path.join(data_dir, "vocal_imitation/references")

        imitation_paths = preprocessing.recursive_wav_paths(imitation_path)
        reference_paths = preprocessing.recursive_wav_paths(reference_path)

        imitation_labels = {}
        for imitation in imitation_paths:
            file_name = os.path.basename(imitation)
            label = file_name[:3]
            imitation_labels[imitation] = label

        reference_labels = {}
        for path in reference_paths:
            category_name = os.path.basename(os.path.dirname(path))
            file_name = os.path.basename(path)
            label = category_name[:3]

            is_canonical = 'perfect' in file_name.lower()
            reference_labels[path] = {"label": label,
                                      "is_canonical": is_canonical}

        preprocessing.calculate_spectrograms(imitation_paths, imitation_labels, 'imitations', 'vocal_imitation',
                                             preprocessing.imitation_spectrogram)
        preprocessing.calculate_spectrograms(reference_paths, reference_labels, 'references', 'vocal_imitation',
                                             preprocessing.reference_spectrogram)

    @staticmethod
    def generate_occurrences(imitation_labels, reference_labels):
        i_s = set(v for v in imitation_labels.values())
        r_s = set(v['label'] for v in reference_labels.values())
        n_canonical = {}
        for l in r_s:
            found = 0
            for k, v in reference_labels.items():
                if v['label'] == l and v['is_canonical']:
                    found += 1
            n_canonical[l] = found
        n_references = {}
        for l in r_s:
            found = 0
            for k, v in reference_labels.items():
                if v['label'] == l:
                    found += 1
            n_references[l] = found
        n_imitations = {}
        for l in r_s:
            found = 0
            for k, v in imitation_labels.items():
                if v == l:
                    found += 1
            n_imitations[l] = found
        print(n_references)
        labels = set(list(i_s) + list(r_s))
        with open("occurrences.csv", 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'n_references', 'n_canonical_references', 'n_imitations'])
            for label in labels:
                writer.writerow([label, n_references[label], n_canonical[label], n_imitations[label]])


if __name__ == "__main__":
    VocalImitation(recalculate_spectrograms=True)
