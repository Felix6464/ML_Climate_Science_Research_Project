import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Custom dataset class for sequence prediction
class MultivariateDataset(Dataset):
    def __init__(self, data, sequence_length, target_length):
        self.data = data
        self.sequence_length = sequence_length
        self.target_length = target_length

    def __len__(self):
        return len(self.data[0]) - self.sequence_length

    def __getitem__(self, idx):
        sequence = np.array([self.data[i][idx:idx + self.sequence_length] for i in range(len(self.data))])
        target = np.array([self.data[i][idx + self.sequence_length + self.target_length] for i in range(len(self.data))])
        return sequence, target

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

# Create a DataLoader for sequence prediction
def create_dataloader(data, batch_size, sequence_length, target_len, shuffle=False):
    dataset = MultivariateDataset(data, sequence_length, target_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


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