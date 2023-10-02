from utilities import *


# Load a PyTorch tensor from a file located at "./synthetic_data/data/lim_integration_200k.pt"
data = torch.load("./synthetic_data/data/lim_integration_200k.pt")

# Normalize the loaded data using a function called 'normalize_data'
data = normalize_data(data)

# Keep only the first 200,000 columns of the data tensor
data = data[:, :200000]

# Define the file path for saving training information
training_info_pth = "final_models_trained/training_info_ffn.txt"
dt = "fnp"

# Define a list 'lr' with a single learning rate value of 0.0001
lr = [0.0001]

# Define a string variable for wandb runs
model_label = "FFN-MODELS"
name = "ffn"

config = {
    "wandb": False,
    "name": name,
    "num_features": 30,
    "hidden_size": 128,
    "input_window": 6,
    "output_window": 1,
    "learning_rate": lr[0],
    "num_layers": 1,
    "num_epochs": 10,
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
               f'./final_models_trained/model_{rand_identifier}.pt')
    print(f"Model saved as model_{rand_identifier}.pt")

    model_dict = {"training_params": config,
                  "models": (rand_identifier, learning_rate)}

    #save_dict(training_info_pth, model_dict)