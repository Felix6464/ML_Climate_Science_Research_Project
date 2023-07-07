import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Custom dataset class for sequence prediction


class TimeSeries(Dataset):
    def __init__(self, xarr, input_window, one_hot_month=False):
        self.input_window = input_window
        self.xarr = xarr

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

# Feedforward network for sequence prediction
class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
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


# Function for training a model
def train(model, dataloader, num_epochs, learning_rate, target_len=1, criterion=nn.MSELoss()):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            inputs = inputs.view(-1, 1).T

            for t in range(target_len):
                torch.autograd.set_detect_anomaly(True)
                print("Input : {} + shape : {} ".format(inputs, inputs.shape))
                outputs = model(inputs)
                print("Output : {} + shape : {} + type : {} ".format(outputs, outputs.shape, type(outputs)))
                #print("Target : {} + shape : {} + type : {} ".format(targets, targets.shape, type(targets)))
                inputs[0, 0:20] = outputs[0, :]
                print("Input : {} + shape : {} ".format(inputs, inputs.shape))


            loss = criterion(outputs, targets)
            #print("Loss : {} + type : {} ".format(loss, type(loss)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        if epoch % 100 == 0:
            losses.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

    return losses


def plot_loss_evolution(losses, num_epochs, learning_rate, hidden_size):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Evolution over Epochs: {num_epochs}\nLearning Rate: {learning_rate}\n, Hidden Size: {hidden_size}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()