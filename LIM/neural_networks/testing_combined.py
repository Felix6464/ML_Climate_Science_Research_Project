from plots import *
from utilities import *
from torch.utils.data import DataLoader
from LIM.LIM_class import *
import torch.nn as nn



def main():
    # Load and normalize data
    data_lim = torch.load("./synthetic_data/data/lim_integration_200k.pt")
    data_lim = normalize_data(data_lim)
    data = torch.load("data_piControl.pt")
    data = normalize_data(data)

    # Calculate the mean and standard deviation along the feature dimension
    print("Data shape : {}".format(data.shape))

    # Specify model numbers for different models to be tested
    model_num_lstm_base = "1122600np"
    model_num_lstm = "6405080np"
    model_num_gru = "1570307np"
    model_num_lstm_input = "2666612np"
    model_num_lstm_input_tf = "42951np"
    model_num_fnn = "6899415fnp"

    # Specify parameters for generating time series data
    input_window = 2
    input_window_ffn = 6
    batch_size = 128
    loss_type = "MSE"

    # Load pre-trained models
    model_lstm_base, model_lstm, model_lstm_inp, model_ffn, model_gru, model_lstm_inp_tf = load_models_testing(
        model_num_lstm_base, model_num_lstm, model_num_lstm_input, model_num_gru, model_num_fnn, model_num_lstm_input_tf
    )

    # Original fit of LIM to synthetic data
    tau = 1
    model_org = LIM(tau)
    model_org.fit(data_lim[:, :100000].numpy())
    model_num_lim = "LIM"
    criterion = nn.MSELoss()

    loss_list = []
    loss_list_temp = []

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    for output_window in x:
        print("Output window : {}".format(output_window))

        # Create DataLoader for testing
        test_dataset = TimeSeriesLSTMnp(data.permute(1, 0), input_window, output_window)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        test_dataset_ffn = TimeSeriesLSTMnp(data.permute(1, 0), input_window_ffn, output_window)
        test_dataloader_ffn = DataLoader(test_dataset_ffn, batch_size=batch_size, shuffle=True, drop_last=True)

        # Evaluate different models
        loss_gru = model_gru.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm_base = model_lstm_base.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm = model_lstm.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm_inp = model_lstm_inp.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm_inp_tf = model_lstm_inp_tf.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_ffn = model_ffn.evaluate_model(test_dataloader_ffn, output_window, batch_size, loss_type)
        print(loss_ffn)

        # Calculate LIM mean forecast for comparison
        loss_lim = 0
        sample_size = len(data[1]) - output_window
        forecast_output = model_org.forecast_mean(data, output_window)
        forecast_output = torch.from_numpy(forecast_output[:, :])

        for datapoint in range(sample_size):
            target = data[:, datapoint + output_window]
            loss_l = criterion(forecast_output[:, datapoint], target)
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

    model_nums = str([model_num_gru, model_num_lstm_base, model_num_lstm, model_num_lstm_input, model_num_lstm_input_tf, model_num_lim])

    # Plot loss for different models and output windows
    plot_loss_horizon_combined(loss_list, model_nums, loss_type, tau=[21, 22, 23])

if __name__ == "__main__":
    main()