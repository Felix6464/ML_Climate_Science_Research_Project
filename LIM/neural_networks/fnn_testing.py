import utilities as ut
from models.FNN_model import *
from LIM.neural_networks.plots import *
from utilities import *
import torch.utils.data as datat
from torch.utils.data import DataLoader



def main():
    # Load a PyTorch tensor from a file located at "./synthetic_data/lim_integration_130k[-1].pt"
    data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")

    # Calculate the mean and standard deviation along the feature dimension using a function called 'normalize_data'
    data = ut.normalize_data(data)

    # Keep only the first 30,000 columns of the data tensor
    data = data[:, :30000]
    print("Data shape : {}".format(data.shape))

    # Calculate the index for splitting the data into training and test sets
    index_train = int(0.9 * len(data[0, :]))

    # Update the 'data' tensor to contain only the data for training (last 10% is used for testing)
    data = data[:, index_train:]

    model_num = ["762324fnp"]
    id = ["FNN"]

    loss_list = []

    for m in model_num:
        saved_model = torch.load(f"./trained_models/ffn/model_{m}.pt")

        # Load the hyperparameters of the model
        params = saved_model["hyperparameters"]

        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        input_window = params["input_window"]
        batch_size = params["batch_size"]
        loss_type = params["loss_type"]

        # Specify the number of features and the stride for generating timeseries raw_data
        num_features = 30
        input_window = 6

        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        losses = []

        for output_window in x:


            input_data_test, target_data_test = dataloader_seq2seq_feat(data,
                                                                        input_window=input_window,
                                                                        output_window=output_window,
                                                                        num_features=num_features)


            # Specify the device to be used for testing
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            input_data_test = torch.from_numpy(input_data_test)
            target_data_test = torch.from_numpy(target_data_test)

            test_dataloader = DataLoader(
                datat.TensorDataset(input_data_test, target_data_test), batch_size=batch_size, shuffle=True, drop_last=True)

            # Initialize the model and load the saved state dict
            model = FeedforwardNetwork(input_size = num_features, hidden_size = hidden_size, output_size=num_features, input_window=input_window)
            model.load_state_dict(saved_model["model_state_dict"])
            model.to(device)


            loss = model.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
            losses.append(loss)
            print(f"Test loss: {loss}")

        loss_list.append((losses, model_num))
    print(f"Test loss: {loss_list}")
    plot_loss_horizon(loss_list, loss_type, id)




if __name__ == "__main__":
    main()
