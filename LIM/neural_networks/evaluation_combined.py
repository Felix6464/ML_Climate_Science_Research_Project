from LIM.neural_networks.train_eval_infrastructure import *

from LIM.utilities.plots import *
from LIM.utilities.utilities import *
from torch.utils.data import DataLoader
from LIM.LIM_class import *
import torch.nn as nn


# Load and normalize synthetic data
data_lim = torch.load("../data/synthetic_data/data_generated/lim_integration_200k.pt")
data_lim = normalize_data(data_lim)

# Load and normalize piControl data
data = torch.load("data_piControl.pt")
data = normalize_data(data)
print("Data shape : {}".format(data.shape))

config = {
    "lstm_base":        "1122600np",
    "lstm":             "6405080np",
    "lstm_input":       "2666612np",
    "lstm_input_tf":    "42951np",
    "gru":              "1570307np",
    "fnn":              "6899415fnp",
    "LIM_model":        LIM(tau=1).fit(data_lim[:, :100000].numpy()),
    "LIM":              "LIM",
    "input_window":     2,
    "input_window_ffn": 6,
    "batch_size":       128,
    "loss_type":        "MSE",
}

# Load pre-trained models
model_lstm_base, model_lstm, model_lstm_inp, model_ffn, model_gru, model_lstm_inp_tf = load_models_testing(
                                                                                                config["lstm_base"],
                                                                                                config["lstm"],
                                                                                                config["lstm_input"],
                                                                                                config["lstm_input_tf"],
                                                                                                config["gru"],
                                                                                                config["fnn"])


loss_list = []
loss_list_temp = []

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

for output_window in x:
    print("Output window : {}".format(output_window))

    # Create DataLoader for testing
    test_dataset = TimeSeriesDatasetnp(data.permute(1, 0), config["input_window"], output_window)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    test_dataset_ffn = TimeSeriesFNNnp(data.permute(1, 0), config["input_window_ffn"])
    test_dataloader_ffn = DataLoader(test_dataset_ffn, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    # Evaluate different models
    loss_gru = evaluate_model(model_gru, test_dataloader, output_window, config["batch_size"], config["loss_type"])
    loss_lstm_base = evaluate_model(model_lstm_base, test_dataloader, output_window, config["batch_size"], config["loss_type"])
    loss_lstm = evaluate_model(model_lstm, test_dataloader, output_window, config["batch_size"], config["loss_type"])
    loss_lstm_inp = evaluate_model(model_lstm_inp, test_dataloader, output_window, config["batch_size"], config["loss_type"])
    loss_lstm_inp_tf = evaluate_model(model_lstm_inp_tf, test_dataloader, output_window, config["batch_size"], config["loss_type"])
    loss_ffn = evaluate_model(model_ffn, test_dataloader_ffn, output_window, config["batch_size"], config["loss_type"])

    # Calculate LIM mean forecast for comparison
    loss_lim = 0
    sample_size = len(data[1]) - output_window
    forecast_output = config["LIM_model"].forecast_mean(data, output_window)
    forecast_output = torch.from_numpy(forecast_output[:, :])

    for datapoint in range(sample_size):
        target = data[:, datapoint + output_window]
        loss_l = nn.MSELoss(forecast_output[:, datapoint], target)
        loss_lim += loss_l.item()

    loss_lim /= sample_size

    loss_list_temp.append([loss_gru, loss_lstm_base, loss_lstm, loss_lstm_inp, loss_lstm_inp_tf, loss_ffn, loss_lim])

# Append loss values for different models to the loss_list
loss_list.append(([lst[0] for lst in loss_list_temp], f"{'GRU'}"))
loss_list.append(([lst[1] for lst in loss_list_temp], f"{'LSTM-Base'}"))
loss_list.append(([lst[2] for lst in loss_list_temp], f"{'LSTM-Enc-Dec'}"))
loss_list.append(([lst[3] for lst in loss_list_temp], f"{'LSTM-Enc-Dec-Input'}"))
loss_list.append(([lst[4] for lst in loss_list_temp], f"{'LSTM-Enc-Dec-Input-TF'}"))
# loss_list.append(([lst[5] for lst in loss_list_temp], f"{'FFN'}"))
loss_list.append(([lst[6] for lst in loss_list_temp], f"{'LIM'}"))

model_nums = str([config["gru"], config["lstm_base"], config["lstm"], config["lstm_input"], config["lstm_input_tf"], config["LIM"]])

# Plot loss for different models and output windows
plot_loss_horizon_combined(loss_list, model_nums, config["loss_type"])
