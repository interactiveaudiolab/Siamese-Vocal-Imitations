import numpy as np

from augmentation.generics import Augmentation


class BackgroundNoiseAugmentation(Augmentation):
    def __init__(self, amplitude):
        super().__init__(replaces=False)
        self.amplitude = amplitude

    def augment(self, audio, sr):
        noise = np.random.normal(0, self.amplitude, len(audio))
        return [np.array(audio) + noise]
