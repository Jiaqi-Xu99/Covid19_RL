import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

size = 200


class CovidDataset(Dataset):
    """loader train data"""

    def __init__(self, csv_file1, csv_file2):
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        self.symptom = pd.read_csv(csv_file1, header=None).dropna()
        self.infection = pd.read_csv(csv_file2, header=None).dropna()
        self.samples = []
        symptom_list = []
        for i in range(0, len(self.symptom), 2):
            symptom = np.array(self.symptom[i:i + 2])
            pad = np.zeros((2, 1))
            symptom = np.c_[pad, symptom]
            symptom_list.append(symptom)

        infection_list = np.array(self.infection)
        for k in range(0, len(infection_list)):
            self.samples.append([symptom_list[k], infection_list[k]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sym, infection = self.samples[idx]
        return sym, infection


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer = nn.Sequential(
            # nn.Conv1d(2, 1, 1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


def train_loop(dataloader, model, loss_fn, optimizer):
    model = model.double()
    for label, (X, y) in enumerate(dataloader):
        # input size (30, 1, 2, 31)
        X = X.view(X.shape[0], 1, 2, 31)
        # Compute prediction and loss
        pred = model(X)
        pred = pred.view(pred.shape[0], 30)
        # pred = pred.squeeze(1)
        # print(pred)
        loss = loss_fn(pred, y.double())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print(f"loss: {loss:>7f}")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.view(X.shape[0], 1, 2, 31)
            pred = model(X)
            # pred = pred.squeeze(1)
            pred = pred.view(pred.shape[0], 30)
            test_loss += loss_fn(pred, y.double()).item()

    test_loss /= num_batches
    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    full_data = CovidDataset('./data_symptom.csv', './data_infection.csv')
    train_data, test_data = torch.utils.data.random_split(full_data, [50000, 10000])
    train_dataloader = DataLoader(train_data, batch_size=30, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=30, shuffle=False)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 500
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), 'model.pth')
