class SiameseDatafile:
    def __init__(self):
        self.train = None
        self.val = None
        self.test = None


class SiamesePartition:
    def __init__(self):
        self.references = []
        self.reference_labels = []
        self.imitations = []
        self.imitation_labels = []
        self.positive_pairs = []
        self.negative_pairs = []
        self.all_pairs = []
        self.labels = []
