import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np



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
        for x in range(0, len(self.symptom), 4):
            symptom = np.array(self.symptom[x:x + 4])
            infection = np.array(self.infection[x:x + 4])
            self.samples.append([symptom, infection])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sym, infection = self.samples[idx]
        return sym, infection


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(4, 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


def train_loop(dataloader, model, loss_fn, optimizer):
    model = model.double()
    for label, (X, y) in enumerate(dataloader):

        # Compute prediction and loss
        print(X)
        pred = model(X)
        loss = loss_fn(pred, y)

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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    full_data = CovidDataset('./data_symptom.csv', './data_infection.csv')
    train_data, test_data = torch.utils.data.random_split(full_data, [5000, 1000])
    train_dataloader = DataLoader(train_data, batch_size=30, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=30, shuffle=False)
    model = NeuralNetwork()
    loss_fn = nn.BCELoss()
    learning_rate = 5e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), 'model_weights.pth')
