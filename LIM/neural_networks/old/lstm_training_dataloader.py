from models.LSTM_enc_dec_try import *
from torch.utils.data import DataLoader
from utilities import *


#data = xr.open_dataarray("./synthetic_data/lim_integration_xarray_130k[-1]q.nc")
data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
#data = torch.load("./data/data_piControl.pt")
data = data[:, :10000]

data = normalize_data(data)

dt = "np"
input_window = 6
output_window = 6
batch_size = 64
one_hot_month = False

#print(data, type(data), data.shape)


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

    train_dataset = TimeSeriesLSTM(train_data, input_window, output_window)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = TimeSeriesLSTM(val_data, input_window, output_window)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

else:

    idx_train = int(len(data[0, :]) * 0.7)
    idx_val = int(len(data[0, :]) * 0.2)

    train_data = data[:, :idx_train]
    val_data = data[:, idx_train: idx_train+idx_val]
    test_data = data[:, idx_train+idx_val: ]

    train_dataset = TimeSeriesLSTMnp(train_data, input_window, output_window)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = TimeSeriesLSTMnp(val_data, input_window, output_window)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)



print("Data shape : {}".format(train_data.shape))

# Setting hyperparameters for training
num_features = 30
hidden_size = 128
num_layers = 2
learning_rate = 0.001
num_epochs = 25
input_window = input_window
output_window = output_window
batch_size = batch_size
training_prediction = "recursive"
teacher_forcing_ratio = 0.4
dynamic_tf = True
shuffle = True
loss_type = "L1"
wind_farm = "britain_time_lag_corr_"
without_input = False

print("Start training")

# Specify the device to be used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM_Sequence_Prediction(input_size = num_features, hidden_size = hidden_size, seq_len=input_window)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss, loss_test = model.train_model(train_dataloader,
                                    val_dataloader,
                                    num_epochs,
                                    input_window,
                                    output_window,
                                    batch_size,
                                    training_prediction,
                                    teacher_forcing_ratio,
                                    learning_rate,
                                    dynamic_tf,
                                    loss_type,
                                    optimizer,
                                    num_features)


rand_identifier = str(np.random.randint(0, 10000000)) + dt
print(f"Model saved as model_{rand_identifier}.pt")

# Save the model and hyperparameters to a file
parameters = {
    'hidden_size': hidden_size,
    "num_layers": num_layers,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs,
    "input_window": input_window,
    "output_window": output_window,
    "batch_size": batch_size,
    "training_prediction": training_prediction,
    "teacher_forcing_ratio": teacher_forcing_ratio,
    "dynamic_tf": dynamic_tf,
    "loss": loss.tolist(),
    "loss_test": loss_test.tolist(),
    "loss_type": loss_type,
    "shuffle": shuffle,
    "wind_farm": wind_farm,
}

torch.save({'hyperparameters': parameters,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
           f'./trained_models/model_{rand_identifier}.pt')