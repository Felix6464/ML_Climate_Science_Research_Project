from LIM.utilities.utilities import *


# Load a PyTorch tensor from a file located at "./synthetic_data/data_generated/lim_integration_200k.pt"
data = torch.load("../data/synthetic_data/data_generated/lim_integration_200k.pt")

# Normalize the loaded data_generated using a function called 'normalize_data'
data = normalize_data(data)

config = {
    "wandb": False,
    "name": "ffn",
    "model_label": "FFN-MODELS",
    "data_type": "fnp",
    "num_features": 30,
    "hidden_size": 128,
    "input_window": 6,
    "output_window": 1,
    "learning_rate": [0.0001],
    "num_layers": 1,
    "num_epochs": 10,
    "batch_size": 64,
    "train_data_len": len(data[0, :]),
    "training_prediction": "recursive",
    "loss_type": "MSE",
    "teacher_forcing_ratio": 0.6,
    "dynamic_tf": True,
    "shuffle": True,
    "one_hot_month": False,
    "training_info_pth": "results/final_models_trained/training_info_ffn.txt",
}


if config["data_type"] == "xr":
    # If using xarray data_generated
    idx_train = int(len(data['time']) * 0.7)
    idx_val = int(len(data['time']) * 0.2)
    print(idx_train, idx_val)

    # Split the data_generated into train, validation, and test sets
    train_data = data[:, :, :idx_train]
    val_data = data[:, :, idx_train: idx_train+idx_val]
    test_data = data[:, :, idx_train+idx_val:]

    # Extract the numpy data_generated from xarray
    train_datan = train_data.data
    val_datan = val_data.data
    test_datan = test_data.data

    # Create DataLoader objects for train and validation datasets
    train_dataset = TimeSeriesFNN(train_data, config["input_window"], config["output_window"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

    val_dataset = TimeSeriesFNN(val_data, config["input_window"], config["output_window"])
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)
else:
    # If using numpy data_generated
    idx_train = int(len(data[0, :]) * 0.7)
    idx_val = int(len(data[0, :]) * 0.2)

    # Split the data_generated into train, validation, and test sets
    train_data = data[:, :idx_train]
    val_data = data[:, idx_train: idx_train+idx_val]
    test_data = data[:, idx_train+idx_val:]

    # Create DataLoader objects for train and validation datasets
    train_dataset = TimeSeriesFNNnp(train_data.permute(1, 0), config["input_window"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

    val_dataset = TimeSeriesFNNnp(val_data.permute(1, 0), config["input_window"])
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

for l in config["learning_rate"]:
    # Loop through different learning rates

    # Set the learning rate for this training run
    learning_rate = l

    print("Start training")

    # Specify the device to be used for training (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Feedforward Network model and move it to the selected device
    model = FeedforwardNetwork(input_size=config["num_features"], hidden_size=config["hidden_size"],
                               output_size=config["num_features"], input_window=config["input_window"])
    model.to(device)

    # Initialize the Adam optimizer with the selected learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Generate a random identifier for this run
    rand_identifier = str(np.random.randint(0, 10000000)) + config["data_type"]
    config["name"] = config["name"] + "-" + rand_identifier

    # Train the model and obtain training and testing losses
    loss, loss_test = model.train_model(train_dataloader, val_dataloader, optimizer, config)

    # Calculate the number of weights and parameters in the model
    num_of_weights = (config["input_window"]*config["hidden_size"] + config["hidden_size"] +
                      config["hidden_size"]*config["output_window"] + config["output_window"])
    num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Update configuration with training results
    config["num_of_weights"] = num_of_weights
    config["num_of_params"] = num_of_params
    config["loss_train"] = loss.tolist()
    config["loss_test"] = loss_test.tolist()
    config["identifier"] = rand_identifier

    # Save the trained model, hyperparameters, and optimizer state
    torch.save({'hyperparameters': config, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               f'results/final_models_trained/model_{rand_identifier}.pt')
    print(f"Model saved as model_{rand_identifier}.pt")

    # Store model information in a dictionary
    model_dict = {"training_params": config, "models": (rand_identifier, learning_rate)}

    #save_dict(training_info_pth, model_dict)