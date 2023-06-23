import itertools
import json
import xarray as xr
import matplotlib.pyplot as plt
import utilities as ut
from LSTM_enc_dec import *


def main():

    data = torch.load("data_piControl.pt")

    # Reshape the data if necessary (assuming a 2D tensor)
    if len(data.shape) == 1:
        data = data.unsqueeze(1)

    # Calculate the mean and standard deviation along the feature dimension
    mean = torch.mean(data, dim=1, keepdim=True)
    std = torch.std(data, dim=1, keepdim=True)

    # Apply normalization using the mean and standard deviation
    data = (data - mean) / std

    index_train = int(0.8 * len(data[0, :]))
    data_test = data[:, index_train:]

    input_window = 6
    output_window = 12

    input_data_test, target_data_test = dataloader_seq2seq(data_test, input_window=input_window, output_window=output_window, num_features=30)

    X_test = torch.from_numpy(input_data_test).type(torch.Tensor)
    Y_test = torch.from_numpy(target_data_test).type(torch.Tensor)

    #Hyperparameters

    hidden_size = 256
    num_layers = 3
    input_window = input_window
    output_window = output_window
    batch_size = 8
    loss_type = "RMSE"

    model_num = 5462805


    print("Start evaluating the model")
    # specify model parameters and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM_seq2seq(input_size = X_test.shape[2], hidden_size = hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(f"./temp_models/model_{model_num}.pt"))
    model.to(device)
    loss_test = model.evaluate_model(X_test, Y_test, input_window, output_window, batch_size, loss_type)

    print("Loss test: {}".format(loss_test))

if __name__ == "__main__":
    main()
