from models.FNN_model import *
import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from utilities import *
import torch.utils.data as datat

#data = xr.open_dataarray("./synthetic_data/lim_integration_xarray_130k[-1]q.nc")
data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
#data = torch.load("./data/data_piControl.pt")
data = data[:, :10000]
data = normalize_data(data)

print("Input data shape", data.shape)

dt = "fnp"
input_window = 6
output_window = 1
batch_size = 64
stride = 1
num_features = 30
one_hot_month = False


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

    train_dataset = TimeSeries(train_data, input_window)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = TimeSeries(val_data, input_window)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

else:

    idx_train = int(len(data[0, :]) * 0.7)
    idx_val = int(len(data[0, :]) * 0.2)

    train_data = data[:, :idx_train]
    val_data = data[:, idx_train: idx_train+idx_val]
    test_data = data[:, idx_train+idx_val: ]

    input_data, target_data = dataloader_seq2seq_feat(train_data,
                                                      input_window=input_window,
                                                      output_window=output_window,
                                                      stride=stride,
                                                      num_features=num_features)

    input_data_val, target_data_val = dataloader_seq2seq_feat(val_data,
                                                          input_window=input_window,
                                                          output_window=output_window,
                                                          stride=stride,
                                                          num_features=num_features)



    # convert windowed data from np.array to PyTorch tensor
    train_data, target_data, val_data, val_target = numpy_to_torch(input_data, target_data, input_data_val, target_data_val)
    print(train_data.shape, target_data.shape, val_data.shape, val_target.shape)
    train_dataloader = DataLoader(
        datat.TensorDataset(train_data, target_data), batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        datat.TensorDataset(val_data, val_target), batch_size=batch_size, shuffle=True, drop_last=True)




#lr = [0.01, 0.001, 0.005, 0.0001, 0.0005, 0.00001]
lr = [0.0001]

for l in lr:

    # Setting hyperparameters for training
    num_features = 30
    hidden_size = 64
    num_layers = 2
    learning_rate = l
    num_epochs = 25
    input_window = input_window
    output_window = output_window
    batch_size = batch_size
    training_prediction = "recursive"
    teacher_forcing_ratio = 0.4
    dynamic_tf = True
    shuffle = True
    loss_type = "L1"

    print("Start training")

    # Specify the device to be used for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FeedforwardNetwork(input_size = num_features, hidden_size = hidden_size, output_size=num_features, input_window=input_window)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss, loss_test = model.train_model(train_dataloader,
                                        val_dataloader,
                                        num_epochs,
                                        learning_rate,
                                        loss_type,
                                        optimizer)


    if True:
        print('Number weights:', (input_window*hidden_size + hidden_size + hidden_size*output_window + output_window))

        print("Number of trainable parameters of our model:",
              sum(p.numel() for p in model.parameters() if p.requires_grad))

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
    }

    torch.save({'hyperparameters': parameters,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               f'./trained_models/ffn/model_{rand_identifier}.pt')