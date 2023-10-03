from models.LSTM_enc_dec import *
from utilities import *
from plots import *


# Load the synthetic data and specify the model number for identification
data = torch.load("./synthetic_data/data/lim_integration_200k.pt")
model_num = "6126861np"

# Calculate the mean and standard deviation along the feature dimension
data = normalize_data(data)  # Normalize the loaded data
data = data[:, 180000:200000]  # Slice the data to a specific range along the feature dimension
num_features = 30  # Specify the number of features in the data

# Load a saved model and its hyperparameters
saved_model = torch.load(f"./final_models_trained/model_{model_num}.pt")  # Load a trained model
params = saved_model["hyperparameters"]  # Get hyperparameters from the saved model
hidden_size = params["hidden_size"]  # Extract the hidden size from hyperparameters
num_layers = params["num_layers"]  # Extract the number of layers from hyperparameters
input_window = params["input_window"]  # Extract the input window size from hyperparameters
output_window = params["output_window"]  # Extract the output window size from hyperparameters
loss_type = params["loss_type"]  # Extract the loss type from hyperparameters
loss = params["loss_train"]  # Extract training loss from hyperparameters
loss_test = params["loss_test"]  # Extract test loss from hyperparameters

# Split the data into training and testing sets
index_train = int(0.8 * len(data[0, :]))  # Calculate the index for splitting the data
data_train = data[:, :index_train]  # Split the data into a training set
data_test = data[:, index_train:]  # Split the data into a testing set

# Prepare data for training and testing
input_data, target_data = dataloader_seq2seq_feat(data_train, input_window=input_window,
                                                  output_window=output_window, num_features=num_features)
input_data_test, target_data_test = dataloader_seq2seq_feat(data_test, input_window=input_window,
                                                            output_window=output_window, num_features=num_features)

print("Data loaded")

# Determine the computing device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert data from NumPy arrays to PyTorch tensors
X_train, Y_train, X_test, Y_test = numpy_to_torch(input_data, target_data, input_data_test, target_data_test)

# Create an LSTM sequence prediction model
model = LSTM_Sequence_Prediction(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers)

# Load the model's trained state dictionary and move the model to the appropriate device
model.load_state_dict(saved_model["model_state_dict"])
model.to(device)

# Plot model forecasts and predictions
plot_model_forecast_PC(model, X_train, Y_train, X_test, Y_test, model_num)

# Print the model's hyperparameters
print("Hyperparameters Model : {}".format(params))

# Plot training and testing loss
plot_loss(loss, loss_test, model_num, "Train_Eval_Loss")
