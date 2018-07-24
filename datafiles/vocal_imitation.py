import csv
import logging
import os

from utils import preprocessing, utils
from utils.utils import get_dataset_dir, zip_shuffle


class VocalImitation:
    def __init__(self, train_ratio, val_ratio, test_ratio, shuffle=True, recalculate_spectrograms=False):
        if recalculate_spectrograms:
            self.calculate_spectrograms()
        logger = logging.getLogger('logger')
        logger.info("train, validation, test ratios = {0}, {1}, {2}".format(train_ratio, val_ratio, test_ratio))

        version = 'vocal_imitation'

        try:
            references = utils.load_npy("references.npy", version)
            reference_labels = utils.load_npy("references_labels.npy", version)

            imitations = utils.load_npy("imitations.npy", version)
            imitation_labels = utils.load_npy("imitations_labels.npy", version)
        except FileNotFoundError:
            logger.warning("No saved spectrograms for vocal imitation. Calculating them...")
            self.calculate_spectrograms()

            references = utils.load_npy("references.npy", version)
            reference_labels = utils.load_npy("references_labels.npy", version)

            imitations = utils.load_npy("imitations.npy", version)
            imitation_labels = utils.load_npy("imitations_labels.npy", version)

        if shuffle:
            references, reference_labels = zip_shuffle(references, reference_labels)
            imitations, imitation_labels = zip_shuffle(imitations, imitation_labels)

    @staticmethod
    def calculate_spectrograms():
        data_dir = get_dataset_dir()
        imitation_path = os.path.join(data_dir, "vocal_imitation/imitations")
        reference_path = os.path.join(data_dir, "vocal_imitation/references")

        csv1 = os.path.join(data_dir, "vocal_imitation/category_name_lookup.csv")
        csv2 = os.path.join(data_dir, "vocal_imitation/ref_filename_lookup.csv")

        with open(csv1) as f1, open(csv2) as f2:
            reader1 = csv.DictReader(f1)
            reader2 = csv.DictReader(f2)
            dict2 = dict((row['new_ref_filename'].replace("_reference.wav", ""), row['old_ref_filename']) for row in reader2)
            canonical_lookup = {}
            for row in reader1:
                if row['new_category_name'] in dict2:
                    canonical_lookup[row['old_category_name']] = dict2[row['new_category_name']]
                else:
                    print(row)
            # canonical_lookup = dict((row['old_category_name'], dict2[row['new_category_name']]) for row in reader1)

        imitation_paths = preprocessing.recursive_wav_paths(imitation_path)
        reference_paths = preprocessing.recursive_wav_paths(reference_path)

        imitation_labels = {}
        for imitation in imitation_paths:
            base_name = os.path.basename(imitation)
            tokens = base_name.split("_")
            label = '_'.join(tokens[2:])  # label is all tokens after the first two
            label = label[:-4]  # trim off file extension
            label = label[:label.rfind('-')]  # trim off number suffix
            imitation_labels[imitation] = label

        reference_labels = {}
        for path in reference_paths:
            base_name = os.path.basename(os.path.dirname(path))
            tokens = base_name.split("_")
            label = '_'.join(tokens[2:])
            if base_name in canonical_lookup:
                is_canonical = canonical_lookup[base_name] == os.path.basename(path)
            else:
                is_canonical = False
                print(base_name)
            reference_labels[path] = {"label": label,
                                      "is_canonical": is_canonical}

        labels = set([v for v in imitation_labels.values()]
                     + [v['label'] for v in reference_labels.values()])

        label_no = dict((v, i) for v, i in enumerate(labels))
        preprocessing.calculate_spectrograms(imitation_paths, imitation_labels, label_no, 'imitations', 'vocal_imitation', preprocessing.imitation_spectrogram)
        preprocessing.calculate_spectrograms(reference_paths, reference_labels, label_no, 'references', 'vocal_imitation', preprocessing.reference_spectrogram)

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
        with open("occurences.csv", 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'n_references', 'n_canonical_references', 'n_imitations'])
            for label in labels:
                writer.writerow([label, n_references[label], n_canonical[label], n_imitations[label]])


if __name__ == "__main__":
    VocalImitation(.35, .15, .5, recalculate_spectrograms=True)
