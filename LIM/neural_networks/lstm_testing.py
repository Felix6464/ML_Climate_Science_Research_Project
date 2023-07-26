from models.LSTM_enc_dec import *
import models.GRU_enc_dec as gru
from LIM.neural_networks.plots.plots import *
from utilities import *
import torch.utils.data as datat
from torch.utils.data import DataLoader


def main():

    #raw_data = torch.load("./synthetic_data/raw_data/lim_integration_TEST_20k[-1]p.pt")
    data = torch.load("./synthetic_data/raw_data/lim_integration_130k[-1].pt")
    data = torch.load("./synthetic_data/raw_data/data_piControl.pt")
    data = normalize_data(data)

    horizon = True

    # Specify the model number of the model to be tested
    # model_num = [("5723466np", "5-5"),
    #              ("6807315np", "5-5-last"),
    #              ("1986764np", "5-5-permute"),
    #              ("2971989np", "5-5-last-only")]
    # model num 2-6
    model_num = [("5415593np", "0.00005_overfit"),
                    ("3417220np", "0.00005_40k"),
                    ("1967542np", "0.00005_60k"),
                    #("7668078np", "0.00005_80k"),
                    #("3737982np", "0.00005_80k_drop"),
                    ("1319995np", "0.00008_control"),
                    ("5797841np", "0.00008_80k")]
                    #("9037420np", "0.00005_norm"),
                    #("4386809np", "0.00005_norm_wod")]
    # model_num = [("8389752np", "12-6"),
    #              ("9177999np", "12-1"),
    #              ("3212319np", "12-2"),
    #              ("8364180np", "6-2"),
    #              ("6113293np", "2-2"),
    #              ("3650104np", "6-1"),
    #              ("9670756np", "2-1"),
    #              ("8221899np", "6-6"),
    #              ("9388021np", "2-6")]
    # #model num min max normalized
    # model_num = [("3237362np", "12-6"),
    #              ("8153371np", "12-1"),
    #              ("1745540np", "12-2"),
    #              ("6881306np", "6-2"),
    #              ("8541071np", "2-2"),
    #              ("9933030np", "6-1"),
    #              ("7583665np", "2-1"),
    #              ("9383950np", "6-6"),
    #              ("6486631np", "2-6")]
    model_num = [("6358986np", "2-6_128_wdecay3")]
    model_num = [("2674237np", "2-6_lstm_64_")]
    model_num = [("1814773np", "12-6"),
                 ("3836957np", "12-1"),
                 ("7623451np", "12-2"),
                 ("7871629np", "6-12"),
                 ("8207731np", "6-4"),
                 ("8234712np", "4-6"),
                 ("228620np", "6-2"),
                 ("4255990np", "2-2"),
                 ("7540476np", "6-1"),
                 ("8087997np", "2-4"),
                 ("2710934np", "2-12"),
                 ("5254853np", "2-1"),
                 ("9001583np", "6-6"),
                 ("3063862np", "2-6")]
    model_num = [("2617592np", "2-6"),
                ("1321170np", "2-8"),
                ("1911386np", "2-10"),
                ("4324123np", "2-12")]
    model_num = [("788373np", "2-10_30k"),
                ("6859043np", "2-10-60k"),
                ("6976427np", "2-10-80k"),
                 ("1027275np", "2-10-XLIM_norm_before"),]
    model_num = [("5390553np", "2-10_XLIM-80k"),
                 ("6976427np", "2-10-80k"),
                 ("9969766np", "2-10-XLIM_tau")]
                 #("6859043np", "2-10-60k"),
                 #("788373np", "2-10_30k")]
    model_num = [("2480517np", "2-10-XL-160k2")]

    id = ["horizon_eval_test23"]

    loss_list = []
    loss_list_eval = []

    for m in range(len(model_num)):
        saved_model = torch.load(f"./trained_models/lstm/model_{model_num[m][0]}.pt")

        # Load the hyperparameters of the model
        params = saved_model["hyperparameters"]
        print("Hyperparameters of model {} : {}".format(model_num[m][0], params))
        wandb.init(project=f"ML-Climate-SST-{'Horizon'}", config=params, name=params['name'])

        hidden_size = params["hidden_size"]
        #dropout = params["dropout"]
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
