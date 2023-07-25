from LIM.neural_networks.models.LSTM_enc_dec import *
from LIM.neural_networks.plots.plots import *
from utilities import *
import torch.utils.data as datat
from torch.utils.data import DataLoader
from LIM.neural_networks.models.LIM_class import *
import torch.nn as nn



def main():

    data_lim = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
    data = torch.load("./synthetic_data/lim_integration_TEST_20k[-1]p.pt")
    data = data_lim[:, 80000:90000]

    # Calculate the mean and standard deviation along the feature dimension
    #data = data_lim[:, 80000:90000]
    print("Data shape : {}".format(data.shape))

    # Specify the model number of the model to be tested
    model_num_lstm_base = "7315929np"
    model_num_lstm = "8365852np"
    model_num_gru = "5492161np"
    model_num_lstm_input = "5322765np"
    model_num_fnn = "905019fnp"

    # Specify the number of features and the stride for generating timeseries data
    input_window = 2
    input_window_ffn = 6
    batch_size = 64
    loss_type = "MSE"

    model_lstm_base, model_lstm, model_lstm_inp, model_ffn, model_gru = load_models_testing(model_num_lstm_base,
                                                                                            model_num_lstm,
                                                                                            model_num_lstm_input,
                                                                                            model_num_gru,
                                                                                            model_num_fnn)

    # original fit of LIM
    tau = 1
    model_org = LIM(tau)
    model_org.fit(data_lim[:, :80000].numpy(), eps=0.01)
    model_num_lim = "LIM"
    criterion = nn.MSELoss()

    loss_list = []

    wandb.init(project=f"ML-Climate-SST-{'Horizon-Combined'}", name="model_comparison")

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    for output_window in x:
        print("Output window : {}".format(output_window))

        test_dataset = lstm.TimeSeriesLSTMnp(data.permute(1, 0),
                                        input_window,
                                        output_window)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=True)

        test_dataset_ffn = lstm.TimeSeriesLSTMnp(data.permute(1, 0),
                                             input_window_ffn,
                                             output_window)

        test_dataloader_ffn = DataLoader(test_dataset_ffn,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True)


        loss_gru = model_gru.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm_base = model_lstm_base.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm = model_lstm.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm_inp = model_lstm_inp.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_ffn = model_ffn.evaluate_model(test_dataloader_ffn, output_window, batch_size, loss_type)


        # LIM mean forecast for comparison
        loss_lim = 0
        sample_size = len(data[1]) - output_window*2

        forecast_output = model_org.forecast(data, [output_window])
        forecast_output = torch.from_numpy(forecast_output[0, :, :])

        for datapoint in range(sample_size):

            target = data[:, datapoint+output_window-1]
            loss_l = criterion(forecast_output[:, datapoint], target)
            loss_lim += loss_l.item()

        loss_lim /= sample_size

        loss_list.append([loss_gru, loss_lstm_base, loss_lstm, loss_lstm_inp, loss_ffn, loss_lim])
        wandb.log({"Loss-Horizon": loss_list})

    loss_list.append(([lst[0] for lst in loss_list], f"{'GRU'}"))
    loss_list.append(([lst[1] for lst in loss_list], f"{'LSTM-Base'}"))
    loss_list.append(([lst[2] for lst in loss_list], f"{'LSTM-Enc-Dec'}"))
    loss_list.append(([lst[3] for lst in loss_list], f"{'LSTM-Enc-Dec-Input'}"))
    loss_list.append(([lst[4] for lst in loss_list], f"{'FFN'}"))
    loss_list.append(([lst[5] for lst in loss_list], f"{'LIM'}"))

    model_nums = str([model_num_gru, model_num_lstm_base, model_num_lstm, model_num_lstm_input, model_num_fnn, model_num_lim])
    plot_loss_horizon_combined(loss_list, model_nums, loss_type)



if __name__ == "__main__":
    main()