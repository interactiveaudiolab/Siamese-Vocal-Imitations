class Augmentation:
    def __init__(self, replaces, *kwargs):
        self.replaces = replaces

    def augment(self, audio, sr):
        raise NotImplementedError
