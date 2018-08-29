import csv
import os

import data_files.utils
from data_files import Datafiles
from utils.utils import get_dataset_dir


class VocalSketch(Datafiles):
    def __init__(self, name):
        super().__init__(name)


class VocalSketch_1_0(VocalSketch):
    def __init__(self):
        super().__init__("vocal_sketch_1_0")

    def prepare_spectrogram_calculation(self):
        data_dir = get_dataset_dir(self.name)
        imitation_path = os.path.join(data_dir, "vocal_imitations/included")
        reference_path = os.path.join(data_dir, "sound_recordings")

        reference_csv = os.path.join(data_dir, "sound_recordings.csv")
        imitation_csv = os.path.join(data_dir, "vocal_imitations.csv")

        with open(reference_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(reference_path, row['filename'])
                self.reference_path_labels[path] = {'label': row['sound_label'],
                                                    'is_canonical': True}

        with open(imitation_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(imitation_path, row['filename'])
                self.imitation_path_labels[path] = row['sound_label']

        self.imitation_paths = data_files.utils.recursive_wav_paths(imitation_path)
        self.reference_paths = data_files.utils.recursive_wav_paths(reference_path)


class VocalSketch_1_1(VocalSketch):
    def __init__(self):
        super().__init__("vocal_sketch_1_1")

    def prepare_spectrogram_calculation(self):
        """
        Calculates normalized imitation and reference spectrograms and saves them as .npy files.
        """
        data_dir = get_dataset_dir(self.name)
        imitation_path_1 = os.path.join(data_dir, "vocal_imitations/included")
        imitation_path_2 = os.path.join(data_dir, "vocal_imitations_set2/included")
        reference_path = os.path.join(data_dir, "sound_recordings")

        imitation_csv_1 = os.path.join(data_dir, "vocal_imitations.csv")
        imitation_csv_2 = os.path.join(data_dir, "vocal_imitaitons_set2.csv")  # not a typo, the CSV's name is misspelled
        reference_csv = os.path.join(data_dir, "sound_recordings.csv")

        with open(reference_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(reference_path, row['filename'])
                self.reference_path_labels[path] = {'label': row['sound_label'],
                                                    'is_canonical': True}

        with open(imitation_csv_1) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(imitation_path_1, row['filename'])
                self.imitation_path_labels[path] = row['sound_label']

        with open(imitation_csv_2) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(imitation_path_2, row['filename'])
                self.imitation_path_labels[path] = row['sound_label']

        self.imitation_paths = data_files.utils.recursive_wav_paths(imitation_path_1) + data_files.utils.recursive_wav_paths(imitation_path_2)
        self.reference_paths = data_files.utils.recursive_wav_paths(reference_path)
