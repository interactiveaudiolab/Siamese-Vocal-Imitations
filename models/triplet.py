import torch

from torch import nn

from models.siamese import Siamese


class Triplet(nn.Module):
    def __init__(self, dropout=True, normalization=True):
        super(Triplet, self).__init__()
        self.siamese = Siamese(dropout=dropout, normalization=normalization)

        linear_layer = nn.Linear(2, 1)

        init_weights = torch.tensor([[50, -50]])
        init_bias = torch.tensor([[0]])

        init_weights = init_weights.float()
        init_bias = init_bias.float()

        init_weights.requires_grad = False
        init_bias.requires_grad = False

        linear_layer.weight = torch.nn.Parameter(init_weights)
        linear_layer.bias = torch.nn.Parameter(init_bias)

        self.final_layer = nn.Sequential(
            linear_layer,
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

    def load_siamese(self, model: nn.Module):
        self.siamese.load_state_dict(model.state_dict())
