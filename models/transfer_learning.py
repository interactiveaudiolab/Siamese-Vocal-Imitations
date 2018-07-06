import torch.nn as nn


class LeftTower(nn.Module):
    def __init__(self, normalization=True):
        super(LeftTower, self).__init__()

        # left branch: vocal imitations
        if normalization:
            # left branch: vocal imitations
            self.left_branch = nn.Sequential(
                nn.Conv2d(1, 48, (6, 6)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(48, 48, (6, 6)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(48, 48, (6, 6)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d((1, 2), (1, 2)),
            )
        else:
            # left branch: vocal imitations
            self.left_branch = nn.Sequential(
                nn.Conv2d(1, 48, (6, 6)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(48, 48, (6, 6)),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(48, 48, (6, 6)),
                nn.ReLU(),
                nn.MaxPool2d((1, 2), (1, 2)),
            )

    def forward(self, left):
        left_output = self.left_branch(left)
        return left_output.view(-1)


class RightTower(nn.Module):
    def __init__(self, normalization=True):
        super(RightTower, self).__init__()
        if normalization:
            self.right_branch = nn.Sequential(
                nn.Conv2d(1, 24, (5, 5)),
                nn.BatchNorm2d(24, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d((2, 4), (2, 4)),

                nn.Conv2d(24, 48, (5, 5)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),
                nn.MaxPool2d((2, 4), (2, 4)),

                nn.Conv2d(48, 48, (5, 5)),
                nn.BatchNorm2d(48, momentum=.01, eps=.001),
                nn.ReLU(),

                nn.Dropout(p=.5),
                nn.Linear(48 * 25 * 2, 64),
                nn.ReLU(),

                nn.Dropout(p=.5),
                nn.Linear(64, 10),
                nn.Softmax()
            )
        else:
            self.right_branch = nn.Sequential(
                nn.Conv2d(1, 24, (5, 5)),
                nn.ReLU(),
                nn.MaxPool2d((2, 4), (2, 4)),

                nn.Conv2d(24, 48, (5, 5)),
                nn.ReLU(),
                nn.MaxPool2d((2, 4), (2, 4)),

                nn.Conv2d(48, 48, (5, 5)),
                nn.ReLU(),

                nn.Dropout(p=.5),
                nn.Linear(48 * 25 * 2, 64),
                nn.ReLU(),

                nn.Dropout(p=.5),
                nn.Linear(64, 10),
                nn.Softmax()
            )

    def forward(self, right):
        right_output = self.right_branch(right)
        return right_output.view(-1)
