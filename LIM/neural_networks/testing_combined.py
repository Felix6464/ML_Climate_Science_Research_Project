from LIM.neural_networks.models.LSTM_enc_dec import *
import models.GRU_enc_dec as gru
import models.LSTM_enc_dec_input as lstm_input
import models.LSTM_enc_dec as lstm
import models.FNN_model as ffn
import models.LSTM as lstm_base
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
    num_features = 30
    input_window = 2
    input_window_ffn = 6
    batch_size = 64
    shuffle = True


    # original fit of LIM
    tau = 1
    model_org = LIM(tau)
    model_org.fit(data_lim[:, :80000].numpy(), eps=0.01)
    model_num_lim = "LIM"
    criterion = nn.MSELoss()

    # Load the saved models
    saved_model_lstm_base = torch.load(f"./trained_models/lstm/model_{model_num_lstm_base}.pt")
    saved_model_lstm = torch.load(f"./trained_models/lstm/model_{model_num_lstm}.pt")
    saved_model_lstm_input = torch.load(f"./trained_models/lstm/model_{model_num_lstm_input}.pt")
    saved_model_fnn = torch.load(f"./trained_models/ffn/model_{model_num_fnn}.pt")
    saved_model_gru = torch.load(f"./trained_models/lstm/model_{model_num_gru}.pt")

    # Load the hyperparameters of the lstm_model base
    params_lb = saved_model_lstm_base["hyperparameters"]

    # Load the hyperparameters of the lstm_model_enc_dec
    params_l = saved_model_lstm["hyperparameters"]

    # Load the hyperparameters of the lstm_input_model_enc_dec
    params_li = saved_model_lstm_input["hyperparameters"]

    # Load the hyperparameters of the fnn_model
    params_f = saved_model_fnn["hyperparameters"]

    # Load the hyperparameters of the fnn_model
    params_g = saved_model_gru["hyperparameters"]


    # Specify the device to be used for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_lstm_base = lstm_base.LSTM_Sequence_Prediction(input_size=num_features,
                                                         hidden_size=params_lb["hidden_size"],
                                                         num_layers=params_lb["num_layers"])

    model_lstm = lstm.LSTM_Sequence_Prediction(input_size=num_features,
                                               hidden_size=params_l["hidden_size"],
                                               num_layers=params_l["num_layers"])

    model_lstm_inp = lstm_input.LSTM_Sequence_Prediction(input_size=num_features,
                                                         hidden_size=params_li["hidden_size"],
                                                         num_layers=params_li["num_layers"])

    model_ffn = ffn.FeedforwardNetwork(input_size=num_features,
                                       hidden_size=params_f["hidden_size"],
                                       output_size=num_features,
                                       input_window=input_window_ffn)

    model_gru = gru.GRU_Sequence_Prediction(input_size=num_features,
                                            hidden_size=params_g["hidden_size"],
                                            num_layers=params_g["num_layers"])

    # Load the saved models
    model_gru.load_state_dict(saved_model_gru["model_state_dict"])
    model_gru = model_gru.to(device)
    model_lstm_base.load_state_dict(saved_model_lstm_base["model_state_dict"])
    model_lstm_base = model_lstm_base.to(device)
    model_lstm.load_state_dict(saved_model_lstm["model_state_dict"])
    model_lstm = model_lstm.to(device)
    model_lstm_inp.load_state_dict(saved_model_lstm_input["model_state_dict"])
    model_lstm_inp = model_lstm_inp.to(device)
    model_ffn.load_state_dict(saved_model_fnn["model_state_dict"])
    model_ffn = model_ffn.to(device)

    loss_list = []

    loss_gru_list = []
    loss_lstm_base_list = []
    loss_lstm_list = []
    loss_lstm_inp_list = []
    loss_ffn_list = []
    loss_lim_list = []

    wandb.init(project=f"ML-Climate-SST-{'Horizon-Combined'}", name="model_comparison")

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    for output_window in x:
        print("Output window : {}".format(output_window))

        test_dataset = lstm.TimeSeriesLSTMnp(data.permute(1, 0),
                                        input_window,
                                        output_window)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     drop_last=True)

        test_dataset_ffn = lstm.TimeSeriesLSTMnp(data.permute(1, 0),
                                             input_window_ffn,
                                             output_window)

        test_dataloader_ffn = DataLoader(test_dataset_ffn,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 drop_last=True)


        loss_gru = model_gru.evaluate_model(test_dataloader, output_window, batch_size, params_g["loss_type"])
        loss_lstm_base = model_lstm_base.evaluate_model(test_dataloader, output_window, batch_size, params_lb["loss_type"])
        loss_lstm = model_lstm.evaluate_model(test_dataloader, output_window, batch_size, params_li["loss_type"])
        loss_lstm_inp = model_lstm_inp.evaluate_model(test_dataloader, output_window, batch_size, params_l["loss_type"])
        loss_ffn = model_ffn.evaluate_model(test_dataloader_ffn, output_window, batch_size, params_f["loss_type"])

        loss_combined = [loss_gru, loss_lstm_base, loss_lstm, loss_lstm_inp, loss_ffn]
        wandb.log({"Loss-Horizon": loss_combined})


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

        loss_gru_list.append(loss_gru)
        loss_lim_list.append(loss_lim)
        loss_lstm_base_list.append(loss_lstm_base)
        loss_lstm_inp_list.append(loss_lstm_inp)
        loss_lstm_list.append(loss_lstm)
        loss_ffn_list.append(loss_ffn)

    loss_list.append((loss_gru_list, f"{'GRU'}"))
    loss_list.append((loss_lstm_base_list, f"{'LSTM-Base'}"))
    loss_list.append((loss_lstm_list, f"{'LSTM-Enc-Dec'}"))
    loss_list.append((loss_lstm_inp_list, f"{'LSTM-Enc-Dec-Input'}"))
    loss_list.append((loss_ffn_list, f"{'FFN'}"))
    loss_list.append((loss_lim_list, f"{'LIM'}"))

    model_nums = str([model_num_gru, model_num_lstm_base, model_num_lstm, model_num_lstm_input, model_num_fnn, model_num_lim])
    plot_loss_horizon_combined(loss_list, model_nums, params_lb["loss_type"])



if __name__ == "__main__":
    main()