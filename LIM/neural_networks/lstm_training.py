import itertools
import json
import xarray as xr
import matplotlib.pyplot as plt
import utilities as ut
from LSTM_enc_dec import *


def main():

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create the DataLoader for first principal component
    data = torch.load("data_piControl.pt")

    # Reshape the data if necessary (assuming a 2D tensor)
    if len(data.shape) == 1:
        data = data.unsqueeze(1)

    # Calculate the mean and standard deviation along the feature dimension
    mean = torch.mean(data, dim=1, keepdim=True)
    std = torch.std(data, dim=1, keepdim=True)

    # Apply normalization using the mean and standard deviation
    data = (data - mean) / std

    index_train = int(0.8 * len(data[0, :]))
    data_train = data[:, :index_train]
    data_test = data[:, index_train:]

    input_window = 6
    output_window = 12

    input_data, target_data = dataloader_seq2seq(data_train, input_window=input_window, output_window=output_window, num_features=30)
    input_data_test, target_data_test = dataloader_seq2seq(data_test, input_window=input_window, output_window=output_window, num_features=30)

    #print("Input data : {} and shape: {} and type : {}".format(input_data, input_data.shape, type(input_data)))

    X_train, Y_train, X_test, Y_test = numpy_to_torch(input_data, target_data, input_data_test, target_data_test)

    #Hyperparameters

    hidden_size = 64
    num_layers = 2
    learning_rate = 0.0001
    num_epochs = 250
    input_window = input_window
    output_window = output_window
    batch_size = 8
    training_prediction = "mixed_teacher_forcing"
    teacher_forcing_ratio = 0.6
    dynamic_tf = True
    shuffle = False
    loss_type = "RMSE"




    print("Start training")
    # specify model parameters and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM_seq2seq(input_size = X_train.shape[2], hidden_size = hidden_size, num_layers=num_layers)
    model.to(device)
    print(device)
    loss = model.train_model(X_train, Y_train, num_epochs, input_window, output_window, batch_size, training_prediction, teacher_forcing_ratio,learning_rate, dynamic_tf, loss_type)


    rand_identifier = np.random.randint(0, 10000000)
    torch.save(model.state_dict(), f'./temp_models/model_{rand_identifier}.pt')

    # Save the model and hyperparameters to a file
    parameters = {
        'hyperparameters': {
            'hidden_size': hidden_size,
            "num_layers": num_layers,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            "input_window": input_window,
            "batch_size": batch_size,
            "training_prediction": training_prediction,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            "dynamic_tf": dynamic_tf,
            "loss_type": loss_type,
            "loss": loss.tolist(),
            "shuffle": shuffle,
        }
    }
    print(f"./temp_models/model_{rand_identifier}_params.txt")
    filename = f"./temp_models/model_{rand_identifier}_params.txt"
    with open(filename, 'w') as f:
        json.dump(parameters, f)

if __name__ == "__main__":
    main()
