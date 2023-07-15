#from LIM.neural_networks.models.LSTM_enc_dec import *
#from LIM.neural_networks.models.LSTM_enc_dec_input import *
#from LIM.neural_networks.models.LSTM import *
from torch.utils.data import DataLoader
from utilities import *
import torch.utils.data as datat
import os


#data = xr.open_dataarray("./synthetic_data/lim_integration_xarray_130k[-1]q.nc")
data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")


data = normalize_data(data)
data = data[:, :10000]
training_info_pth = "trained_models/training_info_lstm.txt"
dt = "np"

num_features = 30
hidden_size = 128
num_layers = 1
num_epochs = 75
batch_size = 64
training_prediction = "recursive"
teacher_forcing_ratio = 0.6
dynamic_tf = True
shuffle = True
loss_type = "MSE"
one_hot_month = False
shuffle = True

lr = [0.01, 0.001, 0.005, 0.0001, 0.0005, 0.00001]
lr = [0.0001]

windows = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10)]
windows = [(6,6)]

for window in windows:


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

        train_dataset = TimeSeriesLSTM(train_data, window[0], window[1])
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_dataset = TimeSeriesLSTM(val_data, window[0], window[1])
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    else:

        idx_train = int(len(data[0, :]) * 0.7)
        idx_val = int(len(data[0, :]) * 0.2)

        train_data = data[:, :idx_train]
        val_data = data[:, idx_train: idx_train+idx_val]
        test_data = data[:, idx_train+idx_val: ]

        input_data, target_data = dataloader_seq2seq_feat(train_data,
                                                          input_window=window[0],
                                                          output_window=window[1],
                                                          num_features=num_features)

        input_data_val, target_data_val = dataloader_seq2seq_feat(val_data,
                                                                    input_window=window[0],
                                                                    output_window=window[1],
                                                                    num_features=num_features)

        # convert windowed data from np.array to PyTorch tensor
        train_data, target_data, val_data, val_target = numpy_to_torch(input_data, target_data, input_data_val, target_data_val)
        train_dataloader = DataLoader(
            datat.TensorDataset(train_data, target_data), batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_dataloader = DataLoader(
            datat.TensorDataset(val_data, val_target), batch_size=batch_size, shuffle=shuffle, drop_last=True)



    for l in lr:
        print("Data shape : {}".format(train_data.shape))

        learning_rate = l

        print("Start training")


        # Specify the device to be used for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #model = LSTM_Sequence_Prediction(input_size = num_features, hidden_size = hidden_size, seq_len=input_window)
        model = LSTM_Sequence_Prediction(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        loss, loss_test = model.train_model(train_dataloader,
                                            val_dataloader,
                                            num_epochs,
                                            window[0],
                                            window[1],
                                            batch_size,
                                            training_prediction,
                                            teacher_forcing_ratio,
                                            learning_rate,
                                            dynamic_tf,
                                            loss_type,
                                            optimizer,
                                            num_features)

        num_of_weigths = (window[0]*hidden_size + hidden_size + hidden_size*window[1] + window[1])
        num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Save the model and hyperparameters to a file
        rand_identifier = str(np.random.randint(0, 10000000)) + dt
        parameters = {
            'hidden_size': hidden_size,
            "num_layers": num_layers,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            "input_window": window[0],
            "output_window": window[1],
            "batch_size": batch_size,
            "training_prediction": training_prediction,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            "dynamic_tf": dynamic_tf,
            "loss": loss.tolist(),
            "loss_test": loss_test.tolist(),
            "loss_type": loss_type,
            "shuffle": shuffle,
            "num_of_weigths": num_of_weigths,
            "num_of_params": num_of_params
        }

        torch.save({'hyperparameters': parameters,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   f'./trained_models/lstm/model_{rand_identifier}.pt')
        print(f"Model saved as model_{rand_identifier}.pt")

        model_dict = {"training_params": [hidden_size,
                                          num_layers,
                                          num_epochs,
                                          window[0],
                                          window[1],
                                          batch_size,
                                          training_prediction,
                                          teacher_forcing_ratio,
                                          dynamic_tf,
                                          loss_type],
                      "models": (rand_identifier, learning_rate)}

    # if os.path.exists(training_info_pth):
    #     # Load the existing dictionary from the file
    #     temp = [load_dictionary(training_info_pth)]
    #     temp.append(model_dict)
    #     save_dictionary(str(temp), training_info_pth)
    # else:
    #     save_dictionary(str(model_dict), training_info_pth)