import librosa

from augmentation.generics import Augmentation


class TimeStretchAugmentation(Augmentation):
    def __init__(self, stretch_multiplier):
        super().__init__(replaces=False)
        self.rate = stretch_multiplier

    def augment(self, audio, sr):
        return [librosa.effects.time_stretch(audio, self.rate)]