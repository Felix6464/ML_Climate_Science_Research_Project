from models.LSTM_enc_dec import *
from torch.utils.data import DataLoader
from utilities import *

#raw_data = xr.open_dataarray("./synthetic_data/lim_integration_xarray_20k[-1]j.nc")
#raw_data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
#data_control = torch.load("./raw_data/data_piControl.pt")
#data_control = normalize_data(data_control)
#raw_data.raw_data = normalize_data(torch.from_numpy(raw_data.raw_data)).numpy()
#print(data_control[0, :10])
#raw_data = torch.cat((data_control, raw_data), 1)
#raw_data = normalize_tensor_individual(raw_data)


data_ = torch.load("./synthetic_data/data/lim_integration_200k.pt")
print(min_max_values_per_slice(data_))
print("Data shape : {}".format(data_.shape))

lr = [0.0005, 0.0001, 0.00005]
lr = [0.0001]

windows = [(2,1), (2,2), (2, 4), (2,6), (2, 10), (2, 12), (4,1), (4, 2), (4, 4), (4, 6), (4, 8), (4, 10), (4, 12),
           (6,1), (6,2), (6,4), (6, 6), (6, 8), (6, 10), (6, 12), (12, 1), (12,2), (12, 6), (12, 8), (12, 10), (12, 12)]
windows = [(2,12), (2,12), (2,12), (2,12), (2,12), (2,12), (2,12), (2,12), (2,12), (2,12)]
windows = [(2,8)]

#cluster test
#data_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000,
              #90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000]

data_sizes = [200000]

model_label = "ENC-DEC-30E-HORIZON"
name = "lstm_enc_dec"
dt = "np"

config = {
    "wandb": True,
    "name": name,
    "num_features": 30,
    "hidden_size": 128,
    "dropout": 0,
    "weight_decay": 0,
    "input_window": windows[0][0],
    "output_window": windows[0][1],
    "learning_rate": lr[0],
    "num_layers": 1,
    "num_epochs": 30,
    "batch_size": 128,
    "train_data_len": len(data_[0, :]),
    "training_prediction": "recursive",
    "loss_type": "MSE",
    "model_label": model_label,
    "teacher_forcing_ratio": 0.5,
    "dynamic_tf": True,
    "shuffle": True,
    "one_hot_month": False,
}
training_info_pth = "trained_models/training_info_lstm.txt"

for window in windows:
    for data_len in data_sizes:
        print("Data size : {} {}".format(data_len, type(data_len)))
        data = data_[:, :data_len]
        data = normalize_data(data)
        print(data.shape)

        config["input_window"] = window[0]
        config["output_window"] = window[1]

        if dt == "xr":

            idx_train = int(len(data['time']) * 0.7)
            idx_val = int(len(data['time']) * 0.2)
            print(idx_train, idx_val)

            train_data = data[: :,  :idx_train]
            val_data = data[: :, idx_train: idx_train+idx_val]
            test_data = data[: :, idx_train+idx_val: ]


            train_dataset = TimeSeriesLSTM(train_data, window[0], window[1])
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=config["batch_size"],
                                          shuffle=config["shuffle"],
                                          drop_last=True)

            val_dataset = TimeSeriesLSTM(val_data, window[0], window[1])
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config["batch_size"],
                                        shuffle=config["shuffle"],
                                        drop_last=True)

        else:

            idx_train = int(len(data[0, :]) * 0.7)
            idx_val = int(len(data[0, :]) * 0.2)

            train_data = data[:, :idx_train]
            val_data = data[:, idx_train: idx_train+idx_val]
            test_data = data[:, idx_train+idx_val: ]


            train_dataset = TimeSeriesLSTMnp(train_data.permute(1, 0),
                                             window[0],
                                             window[1])

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=config["batch_size"],
                                          shuffle=config["shuffle"],
                                          drop_last=True)

            val_dataset = TimeSeriesLSTMnp(val_data.permute(1,0),
                                           window[0],
                                           window[1])

            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config["batch_size"],
                                        shuffle=config["shuffle"],
                                        drop_last=True)


        for l in lr:

            config["learning_rate"] = l

            config["name"] = name + "-" + str(l) + "-" + str(window[0]) + "-" + str(window[1])+ str(data_len)
            #config["name"] = str(window[0]) + "-" + str(window[1]) + str(data_len)

            print("Start training")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = LSTM_Sequence_Prediction(input_size=config["num_features"],
                                             hidden_size=config["hidden_size"],
                                             num_layers=config["num_layers"],
                                             dropout=config["dropout"])
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=l, weight_decay=config["weight_decay"])

            # Save the model and hyperparameters to a file
            rand_identifier = str(np.random.randint(0, 10000000)) + dt
            config["name"] = config["name"] + "-" + rand_identifier


            loss, loss_test = model.train_model(train_dataloader,
                                                val_dataloader,
                                                optimizer,
                                                config)


            num_of_weigths = (window[0]*config["hidden_size"] + config["hidden_size"] + config["hidden_size"]*window[1] + window[1])
            num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            config["num_of_weigths"] = num_of_weigths
            config["num_of_params"] = num_of_params
            config["loss_train"] = loss.tolist()
            config["loss_test"] = loss_test.tolist()
            config["identifier"] = rand_identifier


            torch.save({'hyperparameters': config,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'trained_models_cluster_final/2_8/model_{rand_identifier}.pt')

            print(f"Model saved as model_{rand_identifier}.pt")
            print("Config : {}".format(config))
            wandb.finish()

            model_dict = {"training_params": config,
                          "models": (rand_identifier, l)}

        #save_dict(training_info_pth, model_dict)