import pandas as pd
from LSTM_enc_dec import *
from utilities import *
import json
from plots import plot_train_test_results


# Create the DataLoader for first principal component
data = torch.load("data_piControl.pt")

# Calculate the mean and standard deviation along the feature dimension
mean = torch.mean(data, dim=1, keepdim=True)
std = torch.std(data, dim=1, keepdim=True)

# Apply normalization using the mean and standard deviation
data = (data - mean) / std

index_train = int(0.8 * len(data[0, :]))
data_train = data[:, :index_train]
data_test = data[:, index_train:]

input_window = 6
output_window = 12
hidden_size = 64
num_layers = 2
batch_size = 64

#print("Data_train : {} + shape: {} + type: {}".format(data_train[0], data_train[0].shape, type(data_train)))
input_data, target_data = dataloader_seq2seq(data_train, input_window=input_window, output_window=output_window, num_features=30)
input_data_test, target_data_test = dataloader_seq2seq(data_test, input_window=input_window, output_window=output_window, num_features=30)

# convert windowed data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = numpy_to_torch(input_data, target_data, input_data_test, target_data_test)
model = LSTM_seq2seq(input_size = X_train.shape[2], hidden_size = hidden_size, num_layers=num_layers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


model_num = 988644

model.load_state_dict(torch.load(f"./temp_models/model_{model_num}.pt"))

plot_train_test_results(model, X_train, Y_train, X_test, Y_test, model_num, num_rows=4)



model_params = load_text_as_json(f"./temp_models/model_{model_num}_params.txt")
print(model_params)

plot_loss(model_params["hyperparameters"]['loss'], "train", model_num)
#plot_loss(model_params["hyperparameters"]['loss_test'], model_num)