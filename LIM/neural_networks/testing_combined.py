import utilities as ut
import models.LSTM_enc_dec_input as lstm_input
import models.LSTM_enc_dec as lstm
import models.FNN_model as ffn
from plots import *
from utilities import *


def main():

    data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
    print("Data shape : {}".format(data.shape))

    # Reshape the data if necessary (assuming a 2D tensor)
    if len(data.shape) == 1:
        data = data.unsqueeze(1)

    # Calculate the mean and standard deviation along the feature dimension
    data = ut.normalize_data(data)
    #data = data[:, :5000]

    index_train = int(0.9 * len(data[0, :]))
    data = data[:, index_train:]

    # Specify the model number of the model to be tested
    model_num_lstm = "7416032np"
    model_num_lstm_input = "8619050np"
    model_num_fnn = "3910395fnp"
    #id = [""]

    saved_model_lstm = torch.load(f"./trained_models/lstm/model_{model_num_lstm}.pt")
    saved_model_lstm_input = torch.load(f"./trained_models/lstm/model_{model_num_lstm_input}.pt")
    saved_model_fnn = torch.load(f"./trained_models/ffn/model_{model_num_fnn}.pt")

    # Load the hyperparameters of the lstm_model
    params = saved_model_lstm["hyperparameters"]
    hidden_size_l = params["hidden_size"]
    num_layers_l = params["num_layers"]
    batch_size_l = params["batch_size"]
    loss_type_l = params["loss_type"]

    # Load the hyperparameters of the lstm_input_model
    params = saved_model_lstm_input["hyperparameters"]
    hidden_size_li = params["hidden_size"]
    num_layers_li = params["num_layers"]
    batch_size_li = params["batch_size"]
    loss_type_li = params["loss_type"]

    # Load the hyperparameters of the fnn_model
    params = saved_model_fnn["hyperparameters"]
    hidden_size_f = params["hidden_size"]
    num_layers_f = params["num_layers"]
    batch_size_f = params["batch_size"]
    loss_type_f = params["loss_type"]


    # Specify the number of features and the stride for generating timeseries data
    num_features = 30
    stride = 1
    input_window = 6

    output_windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    loss_lstm_list = []
    loss_lstm_inp_list = []
    loss_ffn_list = []

    for x in range(len(output_windows)):

        input_data_test, target_data_test = dataloader_seq2seq_feat(data,
                                                                    input_window=input_window,
                                                                    output_window=output_windows[x],
                                                                    stride=stride,
                                                                    num_features=num_features)


        # Specify the device to be used for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # convert windowed data from np.array to PyTorch tensor
        X_test = torch.from_numpy(input_data_test)
        Y_test = torch.from_numpy(target_data_test)

        # Initialize the model and load the saved state dict
        model_lstm = lstm.LSTM_Sequence_Prediction(input_size=num_features,
                                                   hidden_size=hidden_size_l,
                                                   num_layers=num_layers_l)
        model_lstm.load_state_dict(saved_model_lstm["model_state_dict"])
        model_lstm.to(device)

        # Initialize the model and load the saved state dict
        model_lstm_inp = lstm_input.LSTM_Sequence_Prediction(input_size=num_features,
                                                             hidden_size=hidden_size_l,
                                                             num_layers=num_layers_l)
        model_lstm_inp.load_state_dict(saved_model_lstm_input["model_state_dict"])
        model_lstm_inp.to(device)

        # Initialize the model and load the saved state dict
        model_ffn = ffn.FeedforwardNetwork(input_size=num_features,
                                           hidden_size=hidden_size_f,
                                           output_size=num_features,
                                           input_window=input_window)
        model_ffn.load_state_dict(saved_model_fnn["model_state_dict"])
        model_ffn.to(device)


        loss_lstm = model_lstm.evaluate_model(X_test, Y_test, output_windows[x], batch_size_l, loss_type_l)
        loss_lstm_inp = model_lstm_inp.evaluate_model(X_test, Y_test, output_windows[x], batch_size_li, loss_type_li)
        loss_ffn = model_ffn.evaluate_model(X_test, Y_test, output_windows[x], batch_size_f, loss_type_f)

        loss_lstm_inp_list.append(loss_lstm_inp)
        loss_lstm_list.append(loss_lstm)
        loss_ffn_list.append(loss_ffn)
        print(f"Test loss LSTM: {loss_lstm}")
        print(f"Test loss LSTM input: {loss_lstm_inp}")
        print(f"Test loss FFN: {loss_ffn}")

    model_nums = str([model_num_lstm, model_num_lstm_input, model_num_fnn])
    plot_loss_horizon_combined(loss_lstm_list, loss_lstm_inp_list, loss_ffn_list, model_nums, loss_type_l)




if __name__ == "__main__":
    main()