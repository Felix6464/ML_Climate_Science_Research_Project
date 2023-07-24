from models.FNN_model import *
import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from utilities import *
import torch.utils.data as datat
import os

#data = xr.open_dataarray("./synthetic_data/lim_integration_xarray_130k[-1]q.nc")
data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
#data = torch.load("./data/data_piControl.pt")
data = data[:, :80000]
data = normalize_data(data)


training_info_pth = "trained_models/training_info_ffn.txt"
dt = "fnp"

lr = [0.01, 0.001, 0.005, 0.0001, 0.0005, 0.00001]
lr = [0.0001]

model_label = "ENC-DEC-INPUT-2-6"
name = "ffn"

config = {
    "wandb": True,
    "name": name,
    "num_features": 30,
    "hidden_size": 128,
    "input_window": 6,
    "output_window": 1,
    "learning_rate": lr[0],
    "num_layers": 1,
    "num_epochs": 50,
    "batch_size": 64,
    "train_data_len": len(data[0, :]),
    "training_prediction": "recursive",
    "loss_type": "MSE",
    "model_label": model_label,
    "teacher_forcing_ratio": 0.6,
    "dynamic_tf": True,
    "shuffle": True,
    "one_hot_month": False,
}


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

    train_dataset = TimeSeries(train_data, config["input_window"], config["output_window"])
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

    val_dataset = TimeSeries(val_data, config["input_window"], config["output_window"])
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

else:

    idx_train = int(len(data[0, :]) * 0.7)
    idx_val = int(len(data[0, :]) * 0.2)

    train_data = data[:, :idx_train]
    val_data = data[:, idx_train: idx_train+idx_val]
    test_data = data[:, idx_train+idx_val: ]

    train_dataset = TimeSeriesnp(train_data.permute(1, 0), config["input_window"])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["batch_size"],
                                  shuffle=config["shuffle"],
                                  drop_last=True)

    val_dataset = TimeSeriesnp(val_data.permute(1, 0), config["input_window"])
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["batch_size"],
                                shuffle=config["shuffle"],
                                drop_last=True)



for l in lr:

    # Setting hyperparameters for training
    learning_rate = l

    print("Start training")

    # Specify the device to be used for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FeedforwardNetwork(input_size = config["num_features"], hidden_size = config["hidden_size"], output_size=config["num_features"], input_window=config["input_window"])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Save the model and hyperparameters to a file
    rand_identifier = str(np.random.randint(0, 10000000)) + dt
    config["name"] = config["name"] + "-" + rand_identifier

    loss, loss_test = model.train_model(train_dataloader,
                                        val_dataloader,
                                        optimizer,
                                        config)


    num_of_weigths = (config["input_window"]*config["hidden_size"] + config["hidden_size"] + config["hidden_size"]*config["output_window"] + config["output_window"])
    num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config["num_of_weigths"] = num_of_weigths
    config["num_of_params"] = num_of_params
    config["loss_train"] = loss.tolist()
    config["loss_test"] = loss_test.tolist()
    config["identifier"] = rand_identifier

    torch.save({'hyperparameters': config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               f'./trained_models/ffn/model_{rand_identifier}.pt')
    print(f"Model saved as model_{rand_identifier}.pt")

    model_dict = {"training_params": config,
                  "models": (rand_identifier, learning_rate)}

    save_dict(training_info_pth, model_dict)