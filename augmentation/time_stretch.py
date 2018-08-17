import librosa

from augmentation.generics import Augmentation


class TimeStretchAugmentation(Augmentation):
    def __init__(self, rate):
        super().__init__(replaces=False)
        self.rate = rate

    def augment(self, audio, sr):
        return [librosa.effects.time_stretch(audio, self.rate)]
