from LIM.utilities.plots import *
from LIM.utilities.utilities import *
from torch.utils.data import DataLoader


# Load data_generated and normalize it
data = torch.load("../data/data_piControl.pt")
data = normalize_data(data)

config = {
    "wandb": False,
    "horizon": True,
    "train_eval_loss": False,
    "id": ["final-horizon"],
    "model_num": [("6126861np", "LSTM")],           # Example: [("model_number", "Model Name")]
}

# Initialize empty lists to store losses
loss_list = []
loss_list_eval = []

# Loop through each model specified by model_num
for m in range(len(config["model_num"])):
    # Load a saved model
    saved_model = torch.load(f"./results/final_models_trained/model_{config['model_num'][m][0]}.pt")

    # Load hyperparameters of the model
    params = saved_model["hyperparameters"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    input_window = params["input_window"]
    batch_size = params["batch_size"]
    loss_type = params["loss_type"]
    shuffle = params["shuffle"]
    loss_eval = params["loss_test"]

    if config["horizon"] is True:
        # Specify the number of features and output windows for generating time series data_generated
        num_features = 30
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        losses = []

        # Loop through different output windows
        for output_window in x:
            # Create a DataLoader for testing
            test_dataset = TimeSeriesLSTMnp(data.permute(1, 0),
                                            input_window,
                                            output_window)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         drop_last=True)

            # Specify the device for testing (CPU or GPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Initialize the model and load the saved state dictionary
            model = LSTM_Sequence_Prediction(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers)
            model.load_state_dict(saved_model["model_state_dict"])
            model.to(device)

            # Evaluate the model and calculate the loss
            loss = model.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
            print("Output window: {}, Loss: {}".format(output_window, loss))
            losses.append(loss)

        # Append losses for this model to the loss_list
        loss_list.append((losses, config['model_num'][m][1]))

    # Append the evaluation loss for this model to the loss_list_eval
    loss_list_eval.append((loss_eval, config['model_num'][m][1]))

# If in horizon mode, plot the loss values
if config['horizon'] is True:
    plot_loss_horizon(loss_list, loss_type, id)

# Optionally, you can also plot the train/evaluation loss curve for the model
if config['train_eval_loss'] is True:
    plot_loss_combined(loss_list_eval, id, loss_type)
