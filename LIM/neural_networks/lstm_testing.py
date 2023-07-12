import utilities as ut
from LIM.neural_networks.old.LSTM_enc_dec_old import *
from plots import *


def main():

    data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
    print("Data shape : {}".format(data.shape))

    # Reshape the data if necessary (assuming a 2D tensor)
    if len(data.shape) == 1:
        data = data.unsqueeze(1)

    # Calculate the mean and standard deviation along the feature dimension
    data = ut.normalize_data(data)

    index_train = int(0.8 * len(data[0, :]))
    data = data[:, index_train:]

    # Specify the model number of the model to be tested
    model_num = "5931219np"
    saved_model = torch.load(f"./trained_models/model_{model_num}.pt")

    # Load the hyperparameters of the model
    params = saved_model["hyperparameters"]

    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    input_window = params["input_window"]
    batch_size = params["batch_size"]
    loss_type = params["loss_type"]

    input_window = 6

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    loss_list = []

    for output_window in x:



        # Specify the number of features and the stride for generating timeseries data
        num_features = 30
        stride = 1

        input_data_test, target_data_test = dataloader_seq2seq_feat(data,
                                                                    input_window=input_window,
                                                                    output_window=output_window,
                                                                    stride=stride,
                                                                    num_features=num_features)


        # Specify the device to be used for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # convert windowed data from np.array to PyTorch tensor
        X_test = torch.from_numpy(input_data_test)
        Y_test = torch.from_numpy(target_data_test)

        # Initialize the model and load the saved state dict
        model = LSTM_Sequence_Prediction(input_size = X_test.shape[2], hidden_size = hidden_size, num_layers=num_layers)
        model.load_state_dict(saved_model["model_state_dict"])
        model.to(device)


        loss = model.evaluate_model(X_test, Y_test, output_window, batch_size, loss_type)
        loss_list.append(loss)
        print(f"Test loss: {loss}")


    print(f"Test loss: {loss_list}")
    plot_loss_horizon(loss_list, model_num, loss_type)




if __name__ == "__main__":
    main()
