import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.dataloader as dataloader

from datasets import VocalImitations
from siamese import Siamese


def main():
    # global parameters
    n_epochs = 30  # 70 in Bongjun's version, 30 in the paper
    model_path = "./model"

    # load up the data
    train_data = dataloader.DataLoader(VocalImitations(), batch_size=128, num_workers=2, shuffle=True)

    # get a siamese network, see Siamese class for architecture
    siamese = Siamese()

    # choose our objective function
    criterion = nn.BCELoss()

    # use stochastic gradient descent, same parameters as in paper
    optimizer = torch.optim.SGD(siamese.parameters(), lr=.01, weight_decay=.0001, momentum=.9, nesterov=True)

    # train the network
    for epoch in range(n_epochs):
        for i, (left, right, labels) in enumerate(train_data):
            labels = labels.float()
            # clear out the gradients
            optimizer.zero_grad()

            # pass a batch through the network
            outputs = siamese(left, right)

            # calculate loss and optimize weights
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f\tpc: %.3f' % (epoch + 1, i + 1, loss.item(), percent_correct(outputs, labels)))

    torch.save(siamese.state_dict(), model_path)

    # evaluate the network
    test_data = dataloader.DataLoader(VocalImitations(is_train=False), batch_size=128, num_workers=2, shuffle=True)
    total_correct = 0
    total = 0
    for i, (left, right, labels) in enumerate(test_data):
        outputs = siamese(left, right)
        total_correct += num_correct(outputs, labels)
        total += labels.shape[0]
        print("Cumulative percent correct on validation data: {0}%".format(100 * total_correct / total_correct))


def percent_correct(outputs, labels):
    correct = num_correct(outputs, labels)
    total = labels.shape[0]
    return correct / total


def num_correct(outputs, labels):
    return torch.sum(torch.round(outputs) == labels).item()


if __name__ == "__main__":
    main()
