import os

import data_files.utils
from data_files import Datafiles
from utils.utils import get_dataset_dir


class VocalImitation(Datafiles):
    def __init__(self):
        super().__init__('vocal_imitation')

    def prepare_spectrogram_calculation(self):
        data_dir = get_dataset_dir(self.name)
        imitation_path = os.path.join(data_dir, "vocal_imitations")
        reference_path = os.path.join(data_dir, "original_recordings")

        imitation_paths = data_files.utils.recursive_wav_paths(imitation_path)
        reference_paths = data_files.utils.recursive_wav_paths(reference_path)

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

        return imitation_labels, imitation_paths, reference_labels, reference_paths
