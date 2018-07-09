import torch.nn as nn


class Tower(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i):
        raise NotImplementedError


class LeftTower(Tower):
    def __init__(self, normalization=True):
        super(LeftTower, self).__init__()
        if normalization:
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

        self.fcn = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(48 * 25 * 2, 64),
            nn.ReLU(),

            nn.Dropout(p=.5),
            nn.Linear(64, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, left):
        left_output = self.right_branch(left)
        left_reshaped = left_output.view(len(left_output), -1)
        output = self.fcn(left_reshaped)
        return output


class RightTower(Tower):
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
                nn.ReLU()
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
            )

        self.fcn = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(48 * 25 * 2, 64),
            nn.ReLU(),

            nn.Dropout(p=.5),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, right):
        right_output = self.right_branch(right)
        right_reshaped = right_output.view(len(right_output), -1)
        output = self.fcn(right_reshaped)
        return output
