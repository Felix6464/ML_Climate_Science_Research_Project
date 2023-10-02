from models.LSTM_enc_dec import *
import models.GRU_enc_dec as gru
from plots import *
from utilities import *
from torch.utils.data import DataLoader


def main():

    data = torch.load("data_piControl.pt")
    data = normalize_data(data)

    horizon = True
    id = ["final-horizon"]

    # Specify the model number of the model to be tested

    ### FINAL PLOTS FOR REPORT
    model_num = [("979173np", "2-1"),
                 ("10428np", "4-1"),
                 ("4832095np", "6-1"),
                 ("6818638np", "12-1")]
    model_num = [("5640620np", "2-2"),
                 ("8561347np", "4-2"),
                 ("9671189np", "6-2"),
                 ("6350493np", "12-2")]
    model_num = [("8893301np", "2-6"),
                 ("3028080np", "4-6"),
                 ("1706285np", "6-6"),
                 ("9624821np", "12-6")]
    model_num = [("527721np", "2-1"),
                 ("9384508np", "2-2"),
                 ("448978np", "2-4"),
                 ("8727459np", "2-6"),
                 ("2083560np", "2-8"),
                 ("5733615np", "2-10"),
                 ("4683225np", "2-12")]
    #DATA
    model_num = [("5405864np", "60k"),
                ("971337np", "50k"),
                ("3363526np", "40k"),
                ("8664232np", "30k"),
                ("6929411np", "20k"),
                ("7753750np", "10k"),
                ("4612028np", "9k"),
                ("822887np", "8k"),
                ("595172np", "7k"),
                ("9429854np", "6k"),
                ("657363np", "5k"),
                ("9065818np", "4k"),
                ("6756506np", "3k"),
                ("1098093np", "2k"),
                ("9289149np", "1k")]
    model_num = [("4683225np", "base"),
                 ("5841502np", "256h"),
                 ("2297907np", "2-layer"),
                 ("1503845np", "0.2 dropout"),
                 ("8602276np", "0.2 dropout +2l")
                 ]


    loss_list = []
    loss_list_eval = []

    for m in range(len(model_num)):
        saved_model = torch.load(f"./final_models_trained/model_{model_num[m][0]}.pt")

        # Load the hyperparameters of the model
        params = saved_model["hyperparameters"]
        #print("Hyperparameters of model {} : {}".format(model_num[m][0], params))
        #wandb.init(project=f"SST-{'SPREAD-Horizon'}", config=params, name=params['name'])

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
                #wandb.log({"Horizon": output_window, "Test Loss": loss})

            loss_list.append((losses, model_num[m][1]))
        loss_list_eval.append((loss_eval, model_num[m][1]))
        #wandb.finish()

    if horizon is True: plot_loss_horizon(loss_list, loss_type, id)
    #plot_loss_combined(loss_list_eval, id, loss_type)




if __name__ == "__main__":
    main()
