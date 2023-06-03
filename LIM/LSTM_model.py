import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib as plt
import torch.optim as optim

# Define the LSTM neural network class
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx):
        input_sequence = self.data[idx:idx+self.sequence_length]
        target = self.data[idx+self.sequence_length]
        return input_sequence, target



# Create a DataLoader for time series data
def create_dataloader(data, sequence_length, batch_size=32, shuffle=False):
    dataset = TimeSeriesDataset(data, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



# Training loop to predict the subsequent float value
def train(model, dataloader, num_epochs, learning_rate, criterion=nn.MSELoss()):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(2))
            #print("Input : {} + shape : {} ".format(inputs, inputs.shape))
            #print("Output : {} + shape : {} ".format(outputs, outputs.shape))
            #print("Target : {} + shape : {} ".format(targets, targets.shape))
            loss = criterion(outputs, targets.unsqueeze(1))
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