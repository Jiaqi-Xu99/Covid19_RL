import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
# import matplotlib.pyplot as plt


# from sklearn.model_selection import KFold

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
        for i in range(0, len(self.symptom), 5):
            symptom = np.array(self.symptom[i:i + 5])
            pad = np.zeros((5, 1))
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
            # nn.Conv1d(in_channels=5, out_channels=1, kernel_size=3),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


def train_loop(dataloader, model, loss_fn, optimizer):
    model = model.double()
    label_train = []
    for X, y in dataloader:
        # input size (30, 1, 3, 31)
        # X = X.view(X.shape[0], 5, 32)
        X = X.view(X.shape[0], 1, 5, 31)
        # Compute prediction and loss
        pred = model(X)
        pred = pred.view(pred.shape[0], 30)
        # pred = pred.squeeze(1)
        loss = loss_fn(pred, y.double())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        label_train.extend(y)
        # print("F1-Score:{:.4f}".format(f1_score(label_train,pred)))
        # print("AUC:{:.4f}".format(roc_auc_score(label_train,pred)))
        # print(f"loss: {loss:>7f}")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X, y in dataloader:
            # X = X.view(X.shape[0], 5, 32)
            X = X.view(X.shape[0], 1, 5, 31)
            pred = model(X)
            # pred = pred.squeeze(1)
            pred = pred.view(pred.shape[0], 30)
            test_loss += loss_fn(pred, y.double()).item()
            y_true.append(y.numpy().flatten())
            y_pred.append(pred.numpy().flatten())
    test_loss /= num_batches
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    print(f"Avg loss: {test_loss:>8f} \n")
    return y_true, y_pred


if __name__ == '__main__':
    full_data = CovidDataset('./data_symptom_size4.csv', './data_infection_size4.csv')
    # cross k-flod
    # kfold = KFold(n_splits=5, shuffle=True)
    # for fold, (train_ids, test_ids) in enumerate(kfold.split(full_data)):
    #     train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    #     test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    #     train_dataloader = DataLoader(full_data, batch_size=30, sampler=train_subsampler)
    #     test_dataloader = DataLoader(full_data, batch_size=30, sampler=test_subsampler)
    # no cross k-flod
    train_data, test_data = torch.utils.data.random_split(full_data, [8400, 3600])
    train_dataloader = DataLoader(train_data, batch_size=30, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=30, shuffle=False)
    model = NeuralNetwork()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    epochs = 250
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        y_true, y_pred = test_loop(test_dataloader, model, loss_fn)
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUC: {auc:4f} \n")

    print("Done!")

    torch.save(model.state_dict(), 'model.pth')
