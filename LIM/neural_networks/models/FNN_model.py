import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import random


# Custom dataset class for sequence prediction
class TimeSeries(Dataset):
    def __init__(self, xarr, input_window, one_hot_month=False):
        self.input_window = input_window
        self.xarr = xarr.compute()

    def __len__(self):
        return len(self.xarr['time']) - self.input_window


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.xarr.isel(time=slice(idx, idx+self.input_window))
        target = self.xarr.isel(time=idx+self.input_window)

        # One hot encoding of month
        idx_month = input.isel(time=-1).time.dt.month.astype(int) - 1
        one_hot_month = np.zeros(12)
        one_hot_month[idx_month] = 1
        one_hot_month = torch.from_numpy(one_hot_month).float()

        input = torch.from_numpy(input.data).float()

        if one_hot_month:
            target = torch.from_numpy(target.data[np.newaxis]).float()
        else:
            target = torch.from_numpy(target.data).float()

        label = {
            'idx_input': torch.arange(idx, idx+self.input_window),
            'idx_target': idx+self.input_window,
            'month': one_hot_month
        }

        return input, target, label


class TimeSeriesnp(Dataset):
    def __init__(self, arr, input_window):
        self.input_window = input_window
        self.arr = arr

    def __len__(self):
        return len(self.arr[0, :]) - self.input_window - 1


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.arr[:, idx:idx+self.input_window].float()
        target = self.arr[:, idx+self.input_window].float()

        label = "not set"

        return input, target, label



class MLP(nn.Module):
    """Auto encoder.

    Args:
        z_dim (int): Dimension of latent space.
        encoder ([type]): Encoder NN.
        decoder ([type]): Decoder NN.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, condition_dim):
        super().__init__()
        self.input_dim, self.output_dim, self.hidden_dim, self.condition_dim = input_dim, output_dim, hidden_dim, condition_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.film = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1 * 2)
        )

    def forward(self, x, month):
        x_hat = self.mlp(x)
        temp = self.film(month)
        gamma, beta = temp.chunk(chunks=2, axis=-1)
        x_hat = gamma * x_hat + beta
        return x_hat


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss
# Feedforward network for sequence prediction
class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_window):
        super(FeedforwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size * input_window, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Function for training a model
    def train_model(self, train_dataloader, eval_dataloader, num_epochs, learning_rate, loss_type, optimizer):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        losses = np.full(num_epochs, np.nan)
        losses_test = np.full(num_epochs, np.nan)

        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()


        with trange(num_epochs) as tr:
            for epoch in tr:
                batch_loss = 0.0
                batch_loss_test = 0.0
                train_len = 0
                eval_len = 0

                for input, target in train_dataloader:
                    train_len += 1
                    self.train()
                    # Set gradients to zero in the beginning of each batch
                    optimizer.zero_grad()
                    input = input.view(input.shape[0], input.shape[2] * input.shape[1])
                    target = target.view(target.shape[2], target.shape[0] , target.shape[1] )
                    # Forward pass
                    #print(input.shape)
                    pred = self.forward(input.to(device))

                    # loss function
                    loss = criterion(target.to(device), pred)

                    # backward prop and optimization
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()

                # Compute average loss for the epoch
                batch_loss /= train_len
                losses[epoch] = batch_loss

                self.eval()
                # For validation no gradients are computed
                with torch.no_grad():
                    for input, target in eval_dataloader:
                            eval_len += 1

                            target = target.view(target.shape[2], target.shape[0] , target.shape[1] )
                            input = input.view(input.shape[0], input.shape[2] * input.shape[1])
                             # Forward pass
                            pred = self.forward(input.to(device))

                            # loss function
                            loss = criterion(target.to(device), pred)
                            batch_loss_test += loss.item()

                    batch_loss_test /= eval_len
                    losses_test[epoch] = batch_loss_test
                    print("Epoch: {0:02d}, Training Loss: {1:.4f}, Test Loss: {2:.4f}".format(epoch, batch_loss, batch_loss_test))

        return losses, losses_test



    def evaluate_model(self, X_test, Y_test, target_len, batch_size, loss_type):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        input_test = X_test.to(device)
        target_test = Y_test.to(device)

        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()

        # Calculate the number of batch iterations
        n_batches_test = input_test.shape[1] // batch_size
        num_batch_test = n_batches_test
        n_batches_test = list(range(n_batches_test))

        random.shuffle(n_batches_test)
        batch_loss_test = 0.0

        for batch_idx in n_batches_test:

            with torch.no_grad():
                self.eval()

                input = input_test[:, batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
                target = target_test[:, batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
                input = input.reshape(input.shape[1], input.shape[0] * input.shape[2])

                for i in range(target_len):

                    input = input.to(device)
                    target = target.to(device)

                    Y_test_pred = self.forward(input.float())
                    Y_test_pred = Y_test_pred.to(device)

                    input = torch.cat((input, Y_test_pred), dim=1)
                    input = input[:, 30:]



            loss_test = criterion(Y_test_pred, target.float())
            batch_loss_test += loss_test.item()

        batch_loss_test /= num_batch_test


        return batch_loss_test