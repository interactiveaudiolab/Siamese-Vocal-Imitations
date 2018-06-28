import torch.nn as nn


class LeftTower(nn.Module):
    def __init__(self):
        super(LeftTower, self).__init__()

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

    def forward(self, left, right):
        left_output = self.left_branch(left)
        return left_output.view(-1)


class RightTower(nn.Module):
    def __init__(self):
        super(RightTower, self).__init__()
        # right branch: reference sounds
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

    def forward(self, left, right):
        right_output = self.right_branch(right)
        return right_output.view(-1)
