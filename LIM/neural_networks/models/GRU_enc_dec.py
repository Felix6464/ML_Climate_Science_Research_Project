from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.init as init
import wandb




class TimeSeriesLSTM(Dataset):
    def __init__(self, xarr, input_window, output_window, one_hot_month=False):
        self.input_window = input_window
        self.output_window = output_window
        self.xarr = xarr.compute()

    def __len__(self):
        return len(self.xarr['time']) - self.input_window - self.output_window - 2


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.xarr.isel(time=slice(idx, idx+self.input_window))
        target = self.xarr.isel(time=slice(idx+self.input_window, idx+self.input_window  + self.output_window))

        # One hot encoding of month
        idx_month = input.isel(time=-1).time.dt.month.astype(int) - 1
        one_hot_month = np.zeros(12)
        one_hot_month[idx_month] = 1
        one_hot_month = torch.from_numpy(one_hot_month).float()

        input = torch.from_numpy(input.data).float()
        if one_hot_month is True:
            target = torch.from_numpy(target.data[np.newaxis]).float()
        else:
            target = torch.from_numpy(target.data).float()

        label = {
            'idx_input': torch.arange(idx, idx+self.input_window),
            'idx_target': torch.arange(idx+self.input_window, idx+self.input_window  + self.output_window),
            'month': one_hot_month
        }

        input = input.reshape(input.shape[1], input.shape[0])
        target = target.reshape(target.shape[1], target.shape[0])

        return input, target, label


class TimeSeriesLSTMnp(Dataset):
    def __init__(self, arr, input_window, output_window):
        self.input_window = input_window
        self.output_window = output_window
        self.arr = arr

    def __len__(self):
        return len(self.arr[:, 0]) - self.input_window - self.output_window - 2


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.arr[idx:idx+self.input_window, :].float()
        target = self.arr[idx+self.input_window:idx+self.input_window  + self.output_window, :].float()

        label = "not set"

        return input, target, label


class TimeSeriesDropout(nn.Module):
    def __init__(self, dropout_prob):
        super(TimeSeriesDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if self.training:
            batch_size, seq_length, input_size = x.size()
            mask = torch.rand(batch_size, seq_length, input_size) >= self.dropout_prob
            mask = mask.to(x.device)
            x = x * mask / (1 - self.dropout_prob)
        return x

class GRU_Encoder(nn.Module):
    """
    Encodes time-series sequence
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        """
        : param input_size:     the number of features in the input_data
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        """

        super(GRU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = TimeSeriesDropout(dropout_prob)
        self.grus = nn.ModuleList()

        for i in range(num_layers):
            input_size = input_size if i == 0 else hidden_size
            self.grus.append(nn.GRU(input_size, hidden_size, batch_first=True))



    def forward(self, x_input, hidden, prediction_type=None, dropout=False):
        """
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        """
        if self.dropout.dropout_prob > 0 and prediction_type != "test":
            dropout = True
            x_input = self.dropout(x_input)

        for i in range(self.num_layers):
            gru_out, hidden = self.grus[i](x_input, hidden)
            if dropout is True:
                x_input = self.dropout(hidden.permute(1, 0, 2))
            else:
                x_input = hidden.permute(1, 0, 2)


        return gru_out, hidden

    def init_hidden(self, batch_size):
        """
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        """

        return init.xavier_normal_(torch.empty(1, batch_size, self.hidden_size))


class GRU_Decoder(nn.Module):
    """
    Decodes hidden state output by encoder
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        """
        : param input_size:     the number of features in the input_data
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        """

        super(GRU_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = TimeSeriesDropout(dropout_prob)

        self.grus = nn.ModuleList()

        for i in range(num_layers):
            self.grus.append(nn.GRU(self.hidden_size, self.hidden_size, batch_first=True))

        self.linear = nn.Linear(self.hidden_size, self.input_size)


    def forward(self,decoder_input, decoder_hidden, outputs=None, target_len=None, prediction_type=None, dropout=False):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence
        '''

        if self.dropout.dropout_prob > 0 and prediction_type != "test":
            dropout = True
            decoder_input = self.dropout(decoder_input)

        # Predict recursively
        for t in range(target_len):
            for i in range(self.num_layers):
                gru_out, decoder_hidden = self.grus[i](decoder_input, decoder_hidden)
                if dropout is True:
                    decoder_input = self.dropout(decoder_hidden.permute(1, 0, 2))
                else:
                    decoder_input = decoder_hidden.permute(1, 0, 2)

            decoder_output = self.linear(gru_out.squeeze(0))
            outputs[:, t, :] = decoder_output[:, 0, :]

        return outputs, decoder_hidden




class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss

class GRU_Sequence_Prediction(nn.Module):
    """
    train LSTM encoder-decoder and make predictions
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        '''

        super(GRU_Sequence_Prediction, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = GRU_Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout_prob=dropout)
        self.decoder = GRU_Decoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout_prob=dropout)

    def train_model(self, train_dataloader, eval_dataloader, optimizer, config):
        """
        Train an LSTM encoder-decoder model.

        :param input_len:
        :param target_test:
        :param input_test:
        :param input_tensor:              Input raw_data with shape (seq_len, # in batch, number features)
        :param target_tensor:             Target raw_data with shape (seq_len, # in batch, number features)
        :param n_epochs:                  Number of epochs
        :param target_len:                Number of values to predict
        :param batch_size:                Number of samples per gradient update
        :param training_prediction:       Type of prediction to make during training ('recursive', 'teacher_forcing', or
                                          'mixed_teacher_forcing'); default is 'recursive'
        :param teacher_forcing_ratio:     Float [0, 1) indicating how much teacher forcing to use when
                                          training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
                                          number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
                                          Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
                                          teacher forcing.
        :param learning_rate:             Float >= 0; learning rate
        :param dynamic_tf:                dynamic teacher forcing reduces the amount of teacher forcing for each epoch
        :return losses:                   Array of loss function for each epoch
        """

        if config["wandb"] is True:
            wandb.init(project=f"ML-Climate-SST-{config['model_label']}", config=config, name=config['name'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)

        # Initialize array to store losses for each epoch
        losses = np.full(config["num_epochs"], np.nan)
        losses_test = np.full(config["num_epochs"], np.nan)

        # Initialize optimizer and criterion
        if config["loss_type"] == 'MSE':
            criterion = nn.MSELoss()
        elif config["loss_type"] == 'L1':
            criterion = nn.L1Loss()
        elif config["loss_type"] == 'RMSE':
            criterion = RMSELoss()



        with trange(config["num_epochs"]) as tr:
            for epoch in tr:
                batch_loss = 0.0
                batch_loss_test = 0.0
                train_len = 0
                eval_len = 0

                for input, target, l in eval_dataloader:
                    eval_len += 1

                    input_eval, target_eval = input, target
                    input_eval = input_eval.to(device)
                    target_eval = target_eval.to(device)

                    with torch.no_grad():
                        self.eval()

                        Y_test_pred = self.predict(input_eval, config["output_window"])
                        Y_test_pred = Y_test_pred.to(device)
                        loss_test = criterion(Y_test_pred, target_eval)
                        batch_loss_test += loss_test.item()

                batch_loss_test /= eval_len
                losses_test[epoch] = batch_loss_test

                for input, target, l in train_dataloader:
                    train_len += 1
                    self.train()

                    input_batch, target_batch = input, target
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)


                    # Initialize outputs tensor
                    outputs = torch.zeros(config["batch_size"], config["output_window"], config["num_features"])
                    outputs = outputs.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Encoder forward pass
                    hidden = self.encoder.init_hidden(input_batch.shape[0])
                    hidden = hidden.to(device)
                    encoder_output, encoder_hidden = self.encoder(input_batch, hidden)


                    decoder_input = encoder_hidden.permute(1, 0, 2)

                    outputs, decoder_hidden = self.decoder(decoder_input,
                                                           hidden,
                                                           outputs,
                                                           config["output_window"])


                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # Backpropagation and weight update
                    loss.backward()
                    optimizer.step()

                # Compute average loss for the epoch
                batch_loss /= train_len
                losses[epoch] = batch_loss

                # Dynamic teacher forcing
                if config["dynamic_tf"] and config["teacher_forcing_ratio"] > 0:
                    config["teacher_forcing_ratio"] -= 0.01

                print("Epoch: {0:02d}, Training Loss: {1:.4f}, Test Loss: {2:.4f}".format(epoch, batch_loss, batch_loss_test))

                # Update progress bar with current loss
                tr.set_postfix(loss_test="{0:.3f}".format(batch_loss_test))

                if config["wandb"] is True:
                    wandb.log({"Epoch": epoch, "Training Loss": batch_loss, "Test Loss": batch_loss_test})
                    wandb.watch(criterion, log="all")

            return losses, losses_test


    def evaluate_model(self, test_dataloader, target_len, batch_size, loss_type):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)

        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()

        eval_len = 0
        batch_loss_test = 0.0

        for input, target, l in test_dataloader:
            eval_len += 1
            self.eval()

            input_batch, target_batch = input, target
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():

                Y_test_pred = self.predict(input_batch.float(), target_len)
                Y_test_pred = Y_test_pred.to(device)
                loss_test = criterion(Y_test_pred[:, -1, :], target_batch[:, -1, :])
                batch_loss_test += loss_test.item()

        batch_loss_test /= eval_len


        return batch_loss_test


    def predict(self, input_tensor, target_len, prediction_type='test'):

        """
        : param input_tensor:      input raw_data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        if prediction_type == 'forecast':
            input_tensor = input_tensor.unsqueeze(1)

        hidden = self.encoder.init_hidden(input_tensor.shape[0])
        hidden = hidden.to(device)
        encoder_output, encoder_hidden = self.encoder(input_tensor, hidden, prediction_type)

        # Initialize outputs tensor
        outputs = torch.zeros(input_tensor.shape[0], target_len, input_tensor.shape[2])
        outputs = outputs.to(device)


        decoder_input = encoder_hidden.permute(1, 0, 2)

        outputs, decoder_hidden = self.decoder(decoder_input,
                                               hidden,
                                               outputs=outputs,
                                               target_len=target_len,
                                               prediction_type=prediction_type)

        if prediction_type == 'forecast':
            outputs = outputs.detach()


        return outputs

