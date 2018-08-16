import numpy as np

from augmentation.generics import Augmentation


class WindowingAugmentation(Augmentation):
    def __init__(self, window_length, hop_size, drop_last=False):
        super().__init__(replaces=True)
        self.window_length = window_length
        self.hop_size = hop_size
        self.drop_last = drop_last

    def augment(self, audio, sr):
        window_samples = self.window_length * sr
        hop_samples = self.hop_size * sr
        audio_samples = audio.shape[0]

        if not self.drop_last:  # pad out the last segment
            pad = np.zeros(window_samples - audio_samples % window_samples)
            audio = np.append(audio, pad)
            audio_samples = audio.shape[0]

        segments = []
        start = 0
        while start + window_samples < audio_samples:
            segments.append(audio[start:start + window_samples])
            start += hop_samples

        return segments