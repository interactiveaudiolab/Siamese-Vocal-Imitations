import numpy as np


class Spectrogram:
    def __init__(self, path):
        self.path = path

    def load(self):
        return np.load(self.path)


class Imitation(Spectrogram):
    def __init__(self, path: str, label: str, index: int):
        super().__init__(path)
        self.label = label
        self.index = index


class Reference(Spectrogram):
    def __init__(self, path: str, label: str, is_canonical: bool, index: int):
        super().__init__(path)
        self.label = label
        self.is_canonical = is_canonical
        self.index = index