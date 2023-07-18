from LIM.neural_networks.models.LSTM_enc_dec import *
#from LIM.neural_networks.models.LSTM_enc_dec_input import *
#from LIM.neural_networks.models.LSTM import *
from torch.utils.data import DataLoader
from utilities import *
import torch.utils.data as datat
import os


#data = xr.open_dataarray("./synthetic_data/lim_integration_xarray_130k[-1]q.nc")
data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")


data = normalize_data(data)
data = data[:, :800]
training_info_pth = "trained_models/training_info_lstm.txt"
dt = "np"

lr = [0.01, 0.001, 0.005, 0.0001, 0.0005, 0.00001]
lr = [0.0001]

windows = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10)]
windows = [(6,6)]

config = {
    "num_features": 30,
    "hidden_size": 128,
    "input_window": windows[0][0],
    "output_window": windows[0][1],
    "learning_rate": lr[0],
    "num_layers": 1,
    "num_epochs": 100,
    "batch_size": 64,
    "training_prediction": "recursive",
    "loss_type": "MSE",
    "model_label": "LSTM_ENC_DEC",
    "teacher_forcing_ratio": 0.4,
    "dynamic_tf": True,
    "shuffle": True,
    "one_hot_month": False,
    "shuffle": True
}

for window in windows:

    config["input_window"] = window[0]
    config["output_window"] = window[1]

    if dt == "xr":

        idx_train = int(len(data['time']) * 0.7)
        idx_val = int(len(data['time']) * 0.2)
        print(idx_train, idx_val)

        train_data = data[: :,  :idx_train]
        val_data = data[: :, idx_train: idx_train+idx_val]
        test_data = data[: :, idx_train+idx_val: ]

        train_datan = train_data.data
        val_datan = val_data.data
        test_datan = test_data.data

        train_dataset = TimeSeriesLSTM(train_data, window[0], window[1])
        train_dataloader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)
        val_dataset = TimeSeriesLSTM(val_data, window[0], window[1])
        val_dataloader = DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

    else:

        idx_train = int(len(data[0, :]) * 0.7)
        idx_val = int(len(data[0, :]) * 0.2)

        train_data = data[:, :idx_train]
        val_data = data[:, idx_train: idx_train+idx_val]
        test_data = data[:, idx_train+idx_val: ]

        input_data, target_data = dataloader_seq2seq_feat(train_data,
                                                          input_window=window[0],
                                                          output_window=window[1],
                                                          num_features=config["num_features"])

        input_data_val, target_data_val = dataloader_seq2seq_feat(val_data,
                                                                    input_window=window[0],
                                                                    output_window=window[1],
                                                                    num_features=config["num_features"])

        # convert windowed data from np.array to PyTorch tensor
        train_data, target_data, val_data, val_target = numpy_to_torch(input_data, target_data, input_data_val, target_data_val)
        train_dataloader = DataLoader(
            datat.TensorDataset(train_data, target_data), batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)
        val_dataloader = DataLoader(
            datat.TensorDataset(val_data, val_target), batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)



    for l in lr:

        config["learning_rate"] = l

        learning_rate = l

        print("Start training")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = LSTM_Sequence_Prediction(input_size=config["num_features"], hidden_size=config["hidden_size"], num_layers=config["num_layers"])
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        loss, loss_test = model.train_model(train_dataloader,
                                            val_dataloader,
                                            optimizer,
                                            config)

        num_of_weigths = (window[0]*config["hidden_size"] + config["hidden_size"] + config["hidden_size"]*window[1] + window[1])
        num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Save the model and hyperparameters to a file
        rand_identifier = str(np.random.randint(0, 10000000)) + dt

        config["num_of_weigths"] = num_of_weigths
        config["num_of_params"] = num_of_params
        config["loss_train"] = loss.tolist()
        config["loss_test"] = loss_test.tolist()
        config["identifier"] = rand_identifier

        torch.save({'hyperparameters': config,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   f'./trained_models/lstm/model_{rand_identifier}.pt')
        print(f"Model saved as model_{rand_identifier}.pt")

        model_dict = {"training_params": config,
                      "models": (rand_identifier, learning_rate)}

    save_dict(training_info_pth, model_dict)