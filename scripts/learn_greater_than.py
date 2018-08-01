import numpy as np
import torch
from torch import nn
from torch.nn import BCELoss
from torch.utils.data import dataset, dataloader


class Model(nn.Module):
    def __init__(self, initialize_to_correct=False, fix_params=False):
        super().__init__()
        linear_layer = nn.Linear(2, 1)

        if initialize_to_correct:
            init_weights = torch.Tensor([[50, -50]])  # values arbitrary, just need to be somewhat large and opposite in magnitude
            init_bias = torch.Tensor([[0]])

            init_weights = init_weights.float()
            init_bias = init_bias.float()

            if fix_params:
                init_weights.requires_grad = False
                init_bias.requires_grad = False

            linear_layer.weight = torch.nn.Parameter(init_weights)
            linear_layer.bias = torch.nn.Parameter(init_bias)

        self.final_layer = nn.Sequential(
            linear_layer,
            nn.Sigmoid()
        )

    def forward(self, a, b):
        a = a.view(len(a), -1)
        b = b.view(len(b), -1)
        c = torch.cat((a, b), dim=1)
        return self.final_layer(c)


class Data(dataset.Dataset):
    def __init__(self):
        self.max_len = 99999999

    def __getitem__(self, index):
        a = np.random.rand()
        b = np.random.rand()
        return a, b, 1 if a > b else 0

    def __len__(self):
        return self.max_len


def train():
    data = Data()
    model = Model()
    model = model.cuda()
    batch_size = 1
    training_data = dataloader.DataLoader(data, batch_size=batch_size)

    criterion = BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)

    for i, (a, b, l) in enumerate(training_data):
        optimizer.zero_grad()

        a = a.float()
        b = b.float()
        l = l.float()

        a = a.cuda()
        b = b.cuda()
        l = l.cuda()

        outputs = model(a, b)
        outputs = outputs.view(-1)

        loss = criterion(outputs, l)
        loss.backward()
        optimizer.step()

        if i % 5000 == 0:
            print(loss.item())
            print(model.state_dict())

        if loss.item() < .01:
            print(loss.item())
            print(model.state_dict())
            print("after {0} examples".format(i * batch_size))
            break


if __name__ == '__main__':
    train()
