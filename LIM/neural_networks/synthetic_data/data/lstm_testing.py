from LIM.neural_networks.models.LSTM_enc_dec import *
import LIM.neural_networks.models.GRU_enc_dec as gru
from LIM.neural_networks.plots import *
from LIM.neural_networks.utilities import *
from torch.utils.data import DataLoader


def main():

    data = torch.load("data_piControl.pt")
    data = normalize_data(data)

    horizon = True

    # Specify the model number of the model to be tested

    model_num = [                 ("366124np", "6-1"),
                                  ("6436104np", "6-2"),
                                  ("5766124np", "6-4"),
                                  ("1716158np", "6-6"),
                                  ("4227347np", "6-8"),
                                  ("6916466np", "6-10"),
                                  ("6031523np", "6-12"),
                                  ("1607761np", "12-1"),
                                  ("3778713np", "12-2"),
                                  ("5483003np", "12-4"),
                                  ("3383730np", "12-6"),
                                  ("36039np", "12-8"),
                                  ("7361308np", "12-10"),
                                  ("3031954np", "12-12")]
    model_num = [("517928np", "2-1"),
                 ("4716746np", "2-2"),
                 ("2482928np", "2-4"),
                 ("7125364np", "2-6"),
                 ("4908365np", "2-10"),
                 ("8049569np", "2-12"),
                 ("791884np", "4-1"),
                 ("5740785np", "4-2"),
                 ("9099984np", "4-4"),
                 ("4151419np", "4-6"),
                 ("5460966np", "4-8"),
                 ("65094np", "4-10"),
                 ("2316936np", "4-12")]


    id = ["test-horizon"]

    loss_list = []
    loss_list_eval = []

    for m in range(len(model_num)):
        saved_model = torch.load(f"LIM/neural_networks/trained_models/lstm/model_{model_num[m][0]}.pt")

        # Load the hyperparameters of the model
        params = saved_model["hyperparameters"]
        print("Hyperparameters of model {} : {}".format(model_num[m][0], params))
        wandb.init(project=f"SST-{'Test-Horizon'}", config=params, name=params['name'])

        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        input_window = params["input_window"]
        batch_size = params["batch_size"]
        loss_type = params["loss_type"]
        shuffle = params["shuffle"]
        loss_eval = params["loss_test"]


        if horizon is True:

            # Specify the number of features and the stride for generating timeseries raw_data
            num_features = 30
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            losses = []

            for output_window in x:

                test_dataset = TimeSeriesLSTMnp(data.permute(1, 0),
                                               input_window,
                                               output_window)

                test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            drop_last=True)

                # Specify the device to be used for testing
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Initialize the model and load the saved state dict
                model = LSTM_Sequence_Prediction(input_size = num_features, hidden_size = hidden_size, num_layers=num_layers)
                model.load_state_dict(saved_model["model_state_dict"])
                model.to(device)

                loss = model.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
                print("Output window: {}, Loss: {}".format(output_window, loss))
                losses.append(loss)
                wandb.log({"Horizon": output_window, "Test Loss": loss})

            loss_list.append((losses, model_num[m][1]))
        loss_list_eval.append((loss_eval, model_num[m][1]))
        wandb.finish()

    if horizon is True: plot_loss_horizon(loss_list, loss_type, id)
    plot_loss_combined(loss_list_eval, id, loss_type)




if __name__ == "__main__":
    main()
