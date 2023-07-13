from models.LSTM_enc_dec import *
from utilities import *
from plots import *


# Create the DataLoader for first principal component
data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
model_num = "9114361np"

# Calculate the mean and standard deviation along the feature dimension
data = normalize_data(data)
data = data[:, :20000]


saved_model = torch.load(f"./trained_models/lstm/model_{model_num}.pt")
params = saved_model["hyperparameters"]

hidden_size = params["hidden_size"]
num_layers = params["num_layers"]
learning_rate = params["learning_rate"]
num_epochs = params["num_epochs"]
input_window = params["input_window"]
output_window = params["output_window"]
batch_size = params["batch_size"]
training_prediction = params["training_prediction"]
teacher_forcing_ratio = params["teacher_forcing_ratio"]
dynamic_tf = params["dynamic_tf"]
shuffle = params["shuffle"]
loss_type = params["loss_type"]
loss = params["loss"]
loss_test = params["loss_test"]


index_train = int(0.8 * len(data[0, :]))
data_train = data[:, :index_train]
data_test = data[:, index_train:]

num_features = 30
stride = 1

input_data, target_data = dataloader_seq2seq_feat(data_train, input_window=input_window, output_window=output_window, stride=stride, num_features=num_features)
input_data_test, target_data_test = dataloader_seq2seq_feat(data_test, input_window=input_window, output_window=output_window, stride=stride, num_features=num_features)
print("Data loaded")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert windowed data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = numpy_to_torch(input_data, target_data, input_data_test, target_data_test)
model = LSTM_Sequence_Prediction(input_size = num_features, hidden_size = hidden_size, num_layers=num_layers)

model.load_state_dict(saved_model["model_state_dict"])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(saved_model['optimizer_state_dict'])

#model.load_state_dict(torch.load(f"./trained_models/model_{model_num}.pt"))

#plot_model_forecast(model, X_train, Y_train, X_test, Y_test, model_num)
plot_model_forecast_PC(model, X_train, Y_train, X_test, Y_test, model_num)


print("Hyperparameters Model : {}".format(params))
plot_loss(loss, loss_test, model_num, "Train_Eval_Loss")
