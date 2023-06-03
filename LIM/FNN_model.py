
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Custom dataset class for sequence prediction
class SequencePredictionDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Remove the last value as it has no target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_value = self.data[idx-1]
        target = self.data[idx]
        return input_value, target

# Feedforward network for sequence prediction
class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedforwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Create a DataLoader for sequence prediction
def create_dataloader(data, batch_size, shuffle=False):
    dataset = SequencePredictionDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# Function for training a model
def train(model, dataloader, num_epochs, learning_rate, criterion=nn.MSELoss()):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1)  # Add an extra dimension for input shape (batch_size, 1)
            outputs = model(inputs)
            #print("Input : {} + shape : {} ".format(inputs, inputs.shape))
            #print("Output : {} + shape : {} ".format(outputs, outputs.shape))
            #print("Target : {} + shape : {} ".format(targets, targets.shape))
            loss = criterion(outputs.squeeze(), targets)
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