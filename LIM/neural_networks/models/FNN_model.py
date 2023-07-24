import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import random
import wandb


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
        return len(self.arr[:, 0]) - self.input_window - 1


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.arr[idx:idx+self.input_window, :].float()
        target = self.arr[idx+self.input_window, :].float()

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
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

# Function for training a model
    def train_model(self, train_dataloader, eval_dataloader, optimizer, config):

        if config["wandb"] is True:
            wandb.init(project=f"ML-Climate-SST-{config['model_label']}", name=config['name'])


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)

        losses = np.full(config["num_epochs"], np.nan)
        losses_test = np.full(config["num_epochs"], np.nan)

        # Initialize optimizer and criterion
        criterion = nn.MSELoss()


        with trange(config["num_epochs"]) as tr:
            for epoch in tr:
                batch_loss = 0.0
                batch_loss_test = 0.0
                train_len = 0
                eval_len = 0

                self.eval()
                # For validation no gradients are computed
                with torch.no_grad():
                    for input, target, l in eval_dataloader:
                        eval_len += 1

                        input = input.view(input.shape[0], input.shape[2] * input.shape[1])

                        # Forward pass
                        pred = self.forward(input.to(device))

                        # loss function
                        loss = criterion(pred, target.to(device))
                        batch_loss_test += loss.item()

                    batch_loss_test /= eval_len
                    losses_test[epoch] = batch_loss_test

                self.train()
                for input, target, l in train_dataloader:
                    train_len += 1
                    # Set gradients to zero in the beginning of each batch


                    optimizer.zero_grad()
                    # Forward pass
                    #print(input.shape)
                    input = input.view(input.shape[0], input.shape[2] * input.shape[1])

                    pred = self.forward(input.to(device))

                    # loss function
                    loss = criterion(pred, target.to(device))

                    # backward prop and optimization
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()

                # Compute average loss for the epoch
                batch_loss /= train_len
                losses[epoch] = batch_loss

                print("Epoch: {0:02d}, Training Loss: {1:.4f}, Test Loss: {2:.4f}".format(epoch, batch_loss, batch_loss_test))

                if config["wandb"] is True:
                    wandb.log({"Epoch": epoch, "Training Loss": batch_loss, "Test Loss": batch_loss_test})
                    wandb.watch(criterion, log="all")

        return losses, losses_test



    def evaluate_model(self, test_dataloader, target_len, batch_size, loss_type):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)


        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()

        eval_len = 0
        batch_loss_test = 0.0

        for input, target, l in test_dataloader:
            eval_len += 1
            self.eval()

            input_batch, target_batch = input, target
            input = input_batch.to(device)
            target = target_batch.to(device)
            input = input.view(input.shape[0], input.shape[2] * input.shape[1])

            with torch.no_grad():

                for i in range(target_len):

                    Y_test_pred = self.forward(input.float())
                    Y_test_pred = Y_test_pred.to(device)

                    #print(Y_test_pred.shape)
                    #print(input.shape)
                    input = torch.cat((input, Y_test_pred), dim=1)
                    input = input[:, 30:]

            #print(input.shape)
            #print(target.shape)
            #print(Y_test_pred.shape)
            last_idx = (target_len-1) * input_batch.shape[2]
            loss_test = criterion(Y_test_pred[:, last_idx:], target[:, -1, last_idx:].float())
            batch_loss_test += loss_test.item()

        batch_loss_test /= eval_len


        return batch_loss_test