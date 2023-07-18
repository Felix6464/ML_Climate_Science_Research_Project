import utilities as ut
import models.LSTM_enc_dec_input as lstm_input
import models.LSTM_enc_dec as lstm
import models.FNN_model as ffn
import models.LSTM as lstm_base
from plots import *
from utilities import *
import torch.utils.data as datat
from torch.utils.data import DataLoader
from LIM_class import *
import torch.nn as nn



def main():

    data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")

    # Calculate the mean and standard deviation along the feature dimension
    data = ut.normalize_data(data)
    data = data[:, 100000:110000]
    print("Data shape : {}".format(data.shape))

    # original fit of LIM
    tau = 1
    model_org = LIM(tau)
    model_org.fit(data[:, :80000].numpy(), eps=0.01)
    model_num_lim = "LIM"
    criterion = nn.MSELoss()

    # Specify the model number of the model to be tested
    model_num_lstm_base = "7874149np"
    model_num_lstm = "4919340np"
    model_num_lstm_input = "8424079np"
    model_num_fnn = "762324fnp"

    # Load the saved models
    saved_model_lstm_base = torch.load(f"./trained_models/lstm/model_{model_num_lstm_base}.pt")
    saved_model_lstm = torch.load(f"./trained_models/lstm/model_{model_num_lstm}.pt")
    saved_model_lstm_input = torch.load(f"./trained_models/lstm/model_{model_num_lstm_input}.pt")
    saved_model_fnn = torch.load(f"./trained_models/ffn/model_{model_num_fnn}.pt")

    # Load the hyperparameters of the lstm_model base
    params = saved_model_lstm_base["hyperparameters"]
    hidden_size_lb = params["hidden_size"]
    num_layers_lb = params["num_layers"]
    loss_type_lb = params["loss_type"]

    # Load the hyperparameters of the lstm_model_enc_dec
    params = saved_model_lstm["hyperparameters"]
    hidden_size_l = params["hidden_size"]
    num_layers_l = params["num_layers"]
    loss_type_l = params["loss_type"]

    # Load the hyperparameters of the lstm_input_model_enc_dec
    params = saved_model_lstm_input["hyperparameters"]
    hidden_size_li = params["hidden_size"]
    num_layers_li = params["num_layers"]
    loss_type_li = params["loss_type"]

    # Load the hyperparameters of the fnn_model
    params = saved_model_fnn["hyperparameters"]
    hidden_size_f = params["hidden_size"]
    loss_type_f = params["loss_type"]


    # Specify the number of features and the stride for generating timeseries data
    num_features = 30
    input_window = 6
    batch_size = 64
    shuffle = True

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    loss_list = []

    loss_lstm_base_list = []
    loss_lstm_list = []
    loss_lstm_inp_list = []
    loss_ffn_list = []
    loss_lim_list = []

    for output_window in x:

        input_data_test, target_data_test = dataloader_seq2seq_feat(data,
                                                                    input_window=input_window,
                                                                    output_window=output_window,
                                                                    num_features=num_features)

        input_data_test = torch.from_numpy(input_data_test)
        target_data_test = torch.from_numpy(target_data_test)
        test_dataloader = DataLoader(
            datat.TensorDataset(input_data_test, target_data_test), batch_size=batch_size, shuffle=shuffle, drop_last=True)


        # Specify the device to be used for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_lstm_base = lstm_base.LSTM_Sequence_Prediction(input_size=num_features,
                                        hidden_size=hidden_size_lb,
                                        num_layers=num_layers_lb)

        model_lstm = lstm.LSTM_Sequence_Prediction(input_size=num_features,
                                                   hidden_size=hidden_size_l,
                                                   num_layers=num_layers_l)

        model_lstm_inp = lstm.LSTM_Sequence_Prediction(input_size=num_features,
                                                             hidden_size=hidden_size_li,
                                                             num_layers=num_layers_li)

        model_ffn = ffn.FeedforwardNetwork(input_size=num_features,
                                           hidden_size=hidden_size_f,
                                           output_size=num_features,
                                           input_window=input_window)



        model_lstm_base.load_state_dict(saved_model_lstm_base["model_state_dict"])
        model_lstm_base = model_lstm_base.to(device)
        model_lstm.load_state_dict(saved_model_lstm["model_state_dict"])
        model_lstm = model_lstm.to(device)
        model_lstm_inp.load_state_dict(saved_model_lstm_input["model_state_dict"])
        model_lstm_inp = model_lstm_inp.to(device)
        model_ffn.load_state_dict(saved_model_fnn["model_state_dict"])
        model_ffn = model_ffn.to(device)

        loss_lstm_base = model_lstm_base.evaluate_model(test_dataloader, output_window, batch_size, loss_type_lb)
        loss_lstm = model_lstm.evaluate_model(test_dataloader, output_window, batch_size, loss_type_li)
        loss_lstm_inp = model_lstm_inp.evaluate_model(test_dataloader, output_window, batch_size, loss_type_l)
        loss_ffn = model_ffn.evaluate_model(test_dataloader, output_window, batch_size, loss_type_f)

        loss_lstm_base_list.append(loss_lstm_base)
        loss_lstm_inp_list.append(loss_lstm_inp)
        loss_lstm_list.append(loss_lstm)
        loss_ffn_list.append(loss_ffn)

    data = data.numpy()
    for output_window in x:
        loss = 0
        sample_size = len(data[1]) - output_window*2

        for datapoint in range(sample_size):

            # Forecast mean using LIM model
            lim_integration, times_ = model_org.noise_integration(data[:, datapoint ], timesteps=output_window, num_comp=30)
            lim_integration = lim_integration.T[:, 1:]

            lim_integration = torch.from_numpy(lim_integration)
            target = torch.from_numpy(data[:, datapoint:datapoint+output_window])
            loss_ = criterion(lim_integration, target)
            loss += loss_.item()

        loss /= sample_size
        loss_lim_list.append(loss)
        #print("Loss of LIM for output window {} : {}".format(output_window, loss))

    loss_list.append((loss_lstm_base_list, f"{'LSTM-Base'}"))
    loss_list.append((loss_lstm_list, f"{'LSTM-Enc-Dec'}"))
    loss_list.append((loss_lstm_inp_list, f"{'LSTM-Enc-Dec-Input'}"))
    loss_list.append((loss_ffn_list, f"{'FFN'}"))
    loss_list.append((loss_lim_list, f"{'LIM'}"))

    model_nums = str([model_num_lstm_base, model_num_lstm, model_num_lstm_input, model_num_fnn, model_num_lim])
    plot_loss_horizon_combined(loss_list, model_nums, loss_type_l)




if __name__ == "__main__":
    main()