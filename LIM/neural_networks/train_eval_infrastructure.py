from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.init as init
import wandb


class TimeSeriesDataset(Dataset):
    def __init__(self, xarr, input_window, output_window, one_hot_month=False):
        self.input_window = input_window
        self.output_window = output_window
        self.xarr = xarr.compute()

    def __len__(self):
        return len(self.xarr['time']) - self.input_window - self.output_window - 2


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.xarr.isel(time=slice(idx, idx+self.input_window))
        target = self.xarr.isel(time=slice(idx+self.input_window, idx+self.input_window  + self.output_window))

        # One hot encoding of month
        idx_month = input.isel(time=-1).time.dt.month.astype(int) - 1
        one_hot_month = np.zeros(12)
        one_hot_month[idx_month] = 1
        one_hot_month = torch.from_numpy(one_hot_month).float()

        input = torch.from_numpy(input.data).float()
        if one_hot_month is True:
            target = torch.from_numpy(target.data[np.newaxis]).float()
        else:
            target = torch.from_numpy(target.data).float()

        label = {
            'idx_input': torch.arange(idx, idx+self.input_window),
            'idx_target': torch.arange(idx+self.input_window, idx+self.input_window  + self.output_window),
            'month': one_hot_month
        }

        input = input.reshape(input.shape[1], input.shape[0])
        target = target.reshape(target.shape[1], target.shape[0])

        return input, target, label


class TimeSeriesDatasetnp(Dataset):
    def __init__(self, arr, input_window, output_window):
        self.input_window = input_window
        self.output_window = output_window
        self.arr = arr

    def __len__(self):
        return len(self.arr[:, 0]) - self.input_window - self.output_window - 2


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.arr[idx:idx+self.input_window, :].float()
        target = self.arr[idx+self.input_window:idx+self.input_window  + self.output_window, :].float()

        label = "not set"

        return input, target, label


class TimeSeriesDropout(nn.Module):
    def __init__(self, dropout_prob):
        super(TimeSeriesDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if self.training:
            batch_size, seq_length, input_size = x.size()
            mask = torch.rand(batch_size, seq_length, input_size) >= self.dropout_prob
            mask = mask.to(x.device)
            x = x * mask / (1 - self.dropout_prob)
        return x


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss


def train_model(model, train_dataloader, eval_dataloader, optimizer, config):
    """
    Train a multivariate neural network model.
    """

    if config["wandb"] is True:
        wandb.init(project=f"SST-{config['model_label']}", config=config, name=config['name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)

    # Initialize array to store losses for each epoch
    losses = np.full(config["num_epochs"], np.nan)
    losses_test = np.full(config["num_epochs"], np.nan)

    # Initialize optimizer and criterion
    if config["loss_type"] == 'MSE':
        criterion = nn.MSELoss()
    elif config["loss_type"] == 'L1':
        criterion = nn.L1Loss()
    elif config["loss_type"] == 'RMSE':
        criterion = RMSELoss()

    with trange(config["num_epochs"]) as tr:
        for epoch in tr:
            batch_loss = 0.0
            batch_loss_test = 0.0
            train_len = 0
            eval_len = 0

            for input, target, l in eval_dataloader:
                eval_len += 1

                input_eval, target_eval = input, target
                input_eval = input_eval.to(device)
                target_eval = target_eval.to(device)

                with torch.no_grad():
                    model.eval()

                    Y_test_pred = model._predict(input_eval, config["output_window"])
                    Y_test_pred = Y_test_pred.to(device)
                    loss_test = criterion(Y_test_pred, target_eval)
                    batch_loss_test += loss_test.item()

            batch_loss_test /= eval_len
            losses_test[epoch] = batch_loss_test

            for input, target, l in train_dataloader:
                train_len += 1
                model.train()

                input_batch, target_batch = input, target
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)


                # Initialize outputs tensor
                outputs = torch.zeros(config["batch_size"], config["output_window"], config["num_features"])
                outputs = outputs.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model._forward(input_batch, outputs, config, target_batch)

                if type(outputs) == tuple:
                    outputs = outputs[0]
                    decoder_hidden = outputs[1]

                loss = criterion(outputs, target_batch)
                batch_loss += loss.item()

                # Backpropagation and weight update
                loss.backward()
                optimizer.step()

            # Compute average loss for the epoch
            batch_loss /= train_len
            losses[epoch] = batch_loss

            # Dynamic teacher forcing
            if config["dynamic_tf"] and config["teacher_forcing_ratio"] > 0:
                config["teacher_forcing_ratio"] -= 0.01

            print("Epoch: {0:02d}, Training Loss: {1:.4f}, Test Loss: {2:.4f}".format(epoch, batch_loss, batch_loss_test))

            # Update progress bar with current loss
            tr.set_postfix(loss_test="{0:.3f}".format(batch_loss_test))

            if config["wandb"] is True:
                wandb.log({"Epoch": epoch, "Training Loss": batch_loss, "Test Loss": batch_loss_test})
                wandb.watch(criterion, log="all")

        return losses, losses_test


def evaluate_model(model, test_dataloader, target_len, batch_size, loss_type):

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
        model.eval()

        input_batch, target_batch = input, target
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():

            Y_test_pred = model._predict(input_batch.float(), target_len)
            Y_test_pred = Y_test_pred.to(device)
            loss_test = criterion(Y_test_pred[:, -1, :], target_batch[:, -1, :])
            batch_loss_test += loss_test.item()

    batch_loss_test /= eval_len

    return batch_loss_test

