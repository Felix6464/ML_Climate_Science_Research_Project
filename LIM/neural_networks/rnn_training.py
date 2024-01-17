from LIM.utilities.utilities import *

# Specify the model to be used
from LIM.models.LSTM_enc_dec_input import LSTM_Sequence_Prediction_Input as Model
from LIM.neural_networks.train_eval_infrastructure import *

# Load data_generated from a file and store it in 'data_'
data_ = torch.load("../data/synthetic_data/data_generated/lim_integration_200k.pt")
print("Data shape : {}".format(data_.shape))


config = {
    "wandb": False,
    "data_type": "np",
    "name": "lstm_enc_dec",
    "data_sizes": [100000],
    "windows": [(2, 12)],
    "num_features": 30,
    "hidden_size": 128,
    "dropout": 0,
    "weight_decay": 0,
    "learning_rate": [0.0001],
    "num_layers": 1,
    "num_epochs": 10,
    "batch_size": 128,
    "train_data_len": len(data_[0, :]),
    "training_prediction": "recursive",
    "loss_type": "MSE",
    "model_label": "LSTM-ENC-DEC",
    "teacher_forcing_ratio": 0.5,
    "dynamic_tf": True,
    "shuffle": True,
    "one_hot_month": False,
}
training_info_pth = "results/final_models_trained/training_info_lstm.txt"

for window in config["windows"]:
    for data_len in config["data_sizes"]:

        # Slice the data_generated to the specified length and normalize it
        data = data_[:, :data_len]
        data = normalize_data(data)
        print("Data Shape: ", data.shape)

        # Configure input and output window sizes based on the 'window' tuple
        config["input_window"] = window[0]
        config["output_window"] = window[1]

        if config["data_type"] == "xr":
            # If using xarray data_generated
            idx_train = int(len(data['time']) * 0.7)
            idx_val = int(len(data['time']) * 0.2)
            print(idx_train, idx_val)

            # Split the data_generated into train, validation, and test sets
            train_data = data[: :,  :idx_train]
            val_data = data[: :, idx_train: idx_train+idx_val]
            test_data = data[: :, idx_train+idx_val: ]

            # Create DataLoader objects for train and validation datasets
            train_dataset = TimeSeriesDataset(train_data, window[0], window[1])
            train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

            val_dataset = TimeSeriesDataset(val_data, window[0], window[1])
            val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

        else:
            # If using numpy data_generated
            idx_train = int(len(data[0, :]) * 0.7)
            idx_val = int(len(data[0, :]) * 0.2)

            # Split the data_generated into train, validation, and test sets
            train_data = data[:, :idx_train]
            val_data = data[:, idx_train: idx_train+idx_val]
            test_data = data[:, idx_train+idx_val: ]
            # Create DataLoader objects for train and validation datasets
            train_dataset = TimeSeriesDatasetnp(train_data.permute(1, 0), window[0], window[1])
            train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

            val_dataset = TimeSeriesDatasetnp(val_data.permute(1,0), window[0], window[1])
            val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)

        for l in config["learning_rate"]:
            # Loop through different learning rates

            config["learning_rate"] = l

            # Define a unique identifier for this training run
            config["name"] = config["name"] + "-" + str(l) + "-" + str(window[0]) + "-" + str(window[1])

            print("Start training")

            # Determine the computing device (GPU if available, otherwise CPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Initialize the model and move it to the selected device
            model = Model(input_size=config["num_features"], hidden_size=config["hidden_size"],
                                             num_layers=config["num_layers"], dropout=config["dropout"])
            model.to(device)

            # Initialize the Adam optimizer with the selected learning rate
            optimizer = torch.optim.Adam(model.parameters(), lr=l, weight_decay=config["weight_decay"])

            # Generate a random identifier for this run
            rand_identifier = str(np.random.randint(0, 10000000)) + config["data_type"]
            config["name"] = config["name"] + "-" + rand_identifier
            print(config["name"])

            # Train the model and obtain training and testing losses
            loss, loss_test = train_model(model, train_dataloader, val_dataloader, optimizer, config)

            # Calculate the number of weights and parameters in the model
            num_of_weights = (window[0]*config["hidden_size"] + config["hidden_size"] + config["hidden_size"]*window[1] + window[1])
            num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Update configuration with training results
            config["num_of_weights"] = num_of_weights
            config["num_of_params"] = num_of_params
            config["loss_train"] = loss.tolist()
            config["loss_test"] = loss_test.tolist()
            config["identifier"] = rand_identifier

            # Save the trained model, hyperparameters, and optimizer state
            torch.save({'hyperparameters': config, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, f'results/final_models_trained/model_{rand_identifier}.pt')

            print(f"Model saved as model_{rand_identifier}.pt")
            print("Config : {}".format(config))

            # Finish logging training results (if using a logging framework like Wandb)
            wandb.finish()

            # Store model information in a dictionary
            model_dict = {"training_params": config, "models": (rand_identifier, l)}

        #save_dict(training_info_pth, model_dict)
