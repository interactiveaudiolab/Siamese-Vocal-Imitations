import torch

from torch import nn

from models.siamese import Siamese
from utils.utils import load_model


class Bisiamese(nn.Module):
    def __init__(self, dropout=True, normalization=True):
        super(Bisiamese, self).__init__()
        self.siamese = Siamese(dropout=dropout, normalization=normalization)
        self.final_layer = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, query, near, far):
        # TODO: we can optimize this by only calculating the left/imitation branch once
        near_output = self.siamese(query, near)
        far_output = self.siamese(query, far)

        near_reshaped = near_output.view(len(near_output), -1)
        far_reshaped = far_output.view(len(far_output), -1)
        concatenated = torch.cat((near_reshaped, far_reshaped), dim=1)

        output = self.final_layer(concatenated)
        return output.view(-1)

    def load_siamese(self, path, use_cuda):
        load_model(self.siamese, path, use_cuda)
