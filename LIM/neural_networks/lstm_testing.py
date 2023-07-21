from models.LSTM_enc_dec import *
from LIM.neural_networks.plots.plots import *
from utilities import *
import torch.utils.data as datat
from torch.utils.data import DataLoader


def main():

    data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
    # Calculate the mean and standard deviation along the feature dimension
    data = normalize_data(data)
    data = data[:, 120000:130000]
    print("Data shape : {}".format(data.shape))

    horizon = True

    # Specify the model number of the model to be tested
    model_num = [("913676np", "1-1"),
                 ("2942352np", "2-2"),
                 ("9683124np", "3-3"),
                 ("5229087np", "4-4"),
                 ("6633299np", "5-5"),
                 ("4181719np", "6-6"),
                 ("6927195np", "7-7"),
                 ("6928194np", "8-8"),
                 ("2731141np", "9-9"),
                 ("8674294np", "10-10"),
                 ("9822140np", "11-11"),
                 ("4770780np", "12-12")]
    model_num = [("3415419np", "1-1"),
                 ("796025np", "2-2"),
                 ("8424079np", "3-3"),
                 ("7059384np", "4-4"),
                 ("3941080np", "5-5"),
                 ("4919340np", "6-6")]
    #model_num = [("6373822np", "7-7"),
    #             ("9672309np", "8-8"),
    #             ("1657797np", "9-9"),
    #             ("7068253np", "10-10"),
    #             ("3454772np", "11-11"),
    #             ("3706716np", "12-12")]
    model_num = [("6752170np", "new_dataloader")]
    id = ["test_dataloader"]

    loss_list = []
    loss_list_eval = []

    for m in range(len(model_num)):
        saved_model = torch.load(f"./trained_models/lstm/model_{model_num[m][0]}.pt")

        # Load the hyperparameters of the model
        params = saved_model["hyperparameters"]

        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        input_window = params["input_window"]
        batch_size = params["batch_size"]
        loss_type = params["loss_type"]
        shuffle = params["shuffle"]
        loss_eval = params["loss_test"]

        if horizon is True:

            # Specify the number of features and the stride for generating timeseries data
            num_features = 30
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            losses = []

            for output_window in x:

                # test_dataset = TimeSeriesLSTMnp(data, input_window, output_window)
                # test_dataloader = DataLoader(
                #     test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

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


                # Initialize the model and load the saved state dict
                model = LSTM_Sequence_Prediction(input_size = num_features, hidden_size = hidden_size, num_layers=num_layers)
                model.load_state_dict(saved_model["model_state_dict"])
                model.to(device)


                loss = model.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
                print("Output window: {}, Loss: {}".format(output_window, loss))
                losses.append(loss)

            print(f"Test loss: {losses[-1]}")
            loss_list.append((losses, model_num[m][1]))
        loss_list_eval.append((loss_eval, model_num[m][1]))

    if horizon is True: plot_loss_horizon(loss_list, loss_type, id)
    plot_loss_combined(loss_list_eval, id, loss_type)




if __name__ == "__main__":
    main()
