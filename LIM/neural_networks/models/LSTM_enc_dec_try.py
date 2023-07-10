from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import trange
import torch
import torch.nn as nn



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

        return input, target, label


class TimeSeriesLSTMnp(Dataset):
    def __init__(self, arr, input_window, output_window):
        self.input_window = input_window
        self.output_window = output_window
        self.arr = arr

    def __len__(self):
        return len(self.arr[0, :]) - self.input_window - self.output_window - 2


    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        input = self.arr[:, idx:idx+self.input_window].float()
        target = self.arr[:, idx+self.input_window:idx+self.input_window  + self.output_window].float()


        label = "not set"

        return input, target, label



class LSTMEncoder(nn.Module):
    def __init__(self,input_size, hidden_size, seq_len):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = LSTMCell(self.input_size, self.hidden_size, seq_len)
        #self.lstm2 = LSTMCell2(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_x, hidden_state):

        h_t, c_t = hidden_state[0], hidden_state[1]
        h_t2, c_t2 = hidden_state[0], hidden_state[1]
        #print("input_x.shape", input_x.shape)
        h_t, c_t = self.lstm1(input_x, h_t, c_t) # initial hidden and cell states
        #h_t2, c_t2 = self.lstm2(h_t, h_t2, c_t2) # new hidden and cell states
        output = self.linear(h_t) # output from the last FC layer

        return output, (h_t, c_t)

    def init_hidden(self, batch_size):
        """
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        """

        return (torch.randn(batch_size, self.hidden_size),
                torch.randn(batch_size, self.hidden_size))


class LSTMDecoder(nn.Module):
    def __init__(self,input_size, hidden_size, seq_len):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.seq_len = seq_len
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = LSTMCell2(self.input_size, self.hidden_size)
        #self.lstm2 = LSTMCell2(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, decoder_input, decoder_hidden, outputs=None, target_batch=None, training_prediction=None, target_len=None, teacher_forcing_ratio=None, prediction_type=None):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence
        '''

        h_t, c_t = decoder_hidden[0], decoder_hidden[1]
        h_t2, c_t2 = decoder_hidden[0], decoder_hidden[1]
        decoder_input = None


        if prediction_type == "test" or prediction_type == "forecast":
            for t in range(target_len):
                h_t, c_t = self.lstm1(decoder_input, h_t, c_t)
                #h_t2, c_t2 = self.lstm2(decoder_input, h_t, c_t)
                output = self.linear(h_t)
                outputs[t] = output
                #decoder_input = output
        else:

            if training_prediction == 'recursive':
                # Predict recursively
                for t in range(target_len):
                    h_t, c_t = self.lstm1(decoder_input, h_t, c_t)
                    #h_t2, c_t2 = self.lstm2(decoder_input, h_t, c_t)
                    output = self.linear(h_t)
                    outputs[t] = output
                    #decoder_input = output

            if training_prediction == 'teacher_forcing':
                # Use teacher forcing
                if random.random() < teacher_forcing_ratio:
                    for t in range(target_len):
                        h_t, c_t = self.lstm1(decoder_input, h_t, c_t)
                        h_t, c_t = self.lstm2(h_t, h_t, c_t)
                        output = self.linear(h_t)
                        outputs[t] = output
                        decoder_input = target_batch[t, :, :]

                # Predict recursively
                else:
                    for t in range(target_len):
                        h_t, c_t = self.lstm1(decoder_input, h_t, c_t)
                        h_t, c_t = self.lstm2(h_t, h_t, c_t)
                        output = self.linear(h_t)
                        outputs[t] = output
                        decoder_input = output

            if training_prediction == 'mixed_teacher_forcing':
                # Predict using mixed teacher forcing
                for t in range(target_len):
                    h_t, c_t = self.lstm1(decoder_input, h_t, c_t)
                    h_t, c_t = self.lstm2(h_t, h_t, c_t)
                    output = self.linear(h_t)
                    outputs[t] = output

                    # Predict with teacher forcing
                    if random.random() < teacher_forcing_ratio:
                        decoder_input = target_batch[t, :, :]

                    # Predict recursively
                    else:
                        decoder_input = output

        return outputs, decoder_hidden

class LSTMCell(nn.Module):
    '''LSTMCell for 1d inputs.'''
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 seq_len: int
                 ):
        """
        Args:
            input_dim (int): Input dimensions
            hidden_dim (int): Hidden dimensions
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(self.input_dim * self.seq_len + self.hidden_dim, self.hidden_dim * 4),
            nn.GroupNorm(num_channels=4*self.hidden_dim, num_groups=4)
        )


    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        '''LSTM forward pass
        Args:
            x (torch.Tensor): Input of shape [sequence_length, batch_size, input_dim]
            h (torch.Tensor): Hidden state
            c (torch.Tensor): Cell state
        '''

        seq_len, batch_size, input_dim = x.shape
        x = x.view(batch_size, seq_len * input_dim)

        z = torch.cat((x, h), dim=1)
        # Concatenate input and hidden state along the sequence length dimension
        z = self.linear(z)


        # Remove the extra dimension for sequence length
        i, f, o, g = z.chunk(chunks=4, dim=1)  # Split the tensor along the feature dimension

        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)

        h = h.squeeze(0)  # Remove the extra dimension for sequence length
        c = c.squeeze(0)  # Remove the extra dimension for sequence length

        return h, c

class LSTMCell2(nn.Module):
    '''LSTMCell for 1d inputs.'''
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int
                 ):
        """
        Args:
            input_dim (int): Input dimensions
            hidden_dim (int): Hidden dimensions
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GroupNorm(num_channels=4*self.hidden_dim, num_groups=4)
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        '''LSTM forward pass
        Args:
            x (torch.Tensor): Input of shape [sequence_length, batch_size, input_dim]
            h (torch.Tensor): Hidden state
            c (torch.Tensor): Cell state
        '''

        # Concatenate input and hidden state along the sequence length dimension
        z = self.linear(h)
        # Remove the extra dimension for sequence length
        i, f, o, g = z.chunk(chunks=4, axis=1)  # Split the tensor along the feature dimension

        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)

        h = h.squeeze(0)  # Remove the extra dimension for sequence length
        c = c.squeeze(0)  # Remove the extra dimension for sequence length

        return h, c



class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss

class LSTM_Sequence_Prediction(nn.Module):
    """
    train LSTM encoder-decoder and make predictions
    """

    def __init__(self, input_size, hidden_size, seq_len):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        '''

        super(LSTM_Sequence_Prediction, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.encoder = LSTMEncoder(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len)
        self.decoder = LSTMDecoder(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len)

    def train_model(self, train_dataloader, eval_dataloader, n_epochs, input_len, target_len, batch_size,
                    training_prediction, teacher_forcing_ratio, learning_rate, dynamic_tf, loss_type, optimizer, num_features):
        """
        Train an LSTM encoder-decoder model.

        :param input_len:
        :param target_test:
        :param input_test:
        :param input_tensor:              Input data with shape (seq_len, # in batch, number features)
        :param target_tensor:             Target data with shape (seq_len, # in batch, number features)
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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        # Initialize array to store losses for each epoch
        losses = np.full(n_epochs, np.nan)
        losses_test = np.full(n_epochs, np.nan)

        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()



        with trange(n_epochs) as tr:
            for epoch in tr:
                batch_loss = 0.0
                batch_loss_test = 0.0

                for batch_idx, train_data in train_dataloader:
                    self.train()

                    input_batch, target_batch = batch_idx, train_data
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)


                    # Initialize outputs tensor
                    outputs = torch.zeros(target_len, batch_size, num_features)
                    outputs = outputs.to(device)

                    # Initialize hidden state for the encoder
                    encoder_hidden = self.encoder.init_hidden(batch_size)
                    encoder_hidden = (encoder_hidden[0].to(device), encoder_hidden[1].to(device))

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Encoder forward pass
                    input_batch = input_batch.view(input_batch.shape[2], input_batch.shape[0] , input_batch.shape[1] )
                    encoder_output, encoder_hidden = self.encoder(input_batch, encoder_hidden)

                    # Decoder input for the current batch
                    decoder_input = input_batch[-1, :, :]
                    decoder_hidden = encoder_hidden


                    outputs, decoder_hidden = self.decoder(decoder_input,
                                                           decoder_hidden,
                                                           outputs,
                                                           target_batch,
                                                           training_prediction,
                                                           target_len,
                                                           teacher_forcing_ratio)

                    target_batch = target_batch.view(target_batch.shape[2], target_batch.shape[0] , target_batch.shape[1])

                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # Backpropagation and weight update
                    loss.backward()
                    optimizer.step()

                # Compute average loss for the epoch
                losses[epoch] = batch_loss

                # Dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio -= 0.01

                for batch_idx, val_data in eval_dataloader:

                    input_eval = batch_idx
                    target_eval = val_data

                    input_eval = input_eval.view(input_eval.shape[2], input_eval.shape[0] , input_eval.shape[1] )
                    input_eval = input_eval.to(device)
                    target_eval = target_eval.to(device)
                    target_eval = target_eval.view(target_eval.shape[2], target_eval.shape[0], target_eval.shape[1])

                    with torch.no_grad():
                        self.eval()

                        Y_test_pred = self.predict(input_eval, target_len)
                        Y_test_pred = Y_test_pred.to(device)
                        loss_test = criterion(Y_test_pred, target_eval)
                        batch_loss_test += loss_test.item()

                losses_test[epoch] = batch_loss_test
                print("Epoch: {0:02d}, Training Loss: {1:.4f}, Test Loss: {2:.4f}".format(epoch, batch_loss, batch_loss_test))

                # Update progress bar with current loss
                tr.set_postfix(loss_test="{0:.3f}".format(batch_loss_test))

            return losses, losses_test


    def evaluate_model(self, test_dataloader, target_len, batch_size, loss_type):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()

        batch_loss_test = 0.0

        for batch_idx, data in enumerate(test_dataloader):
            input_eval, target_eval, label = data
            input_eval = input_eval.to(device)
            target_eval = target_eval.to(device)

            with torch.no_grad():
                self.eval()

                Y_test_pred = self.predict(input_eval, target_len=target_len)
                Y_test_pred = Y_test_pred.to(device)
                loss_test = criterion(Y_test_pred, target_eval)
                batch_loss_test += loss_test.item()

        batch_loss_test = batch_loss_test / len(test_dataloader)



        return batch_loss_test


    def predict(self, input_tensor, target_len, prediction_type='test'):

        """
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        # encode input_tensor
        if prediction_type == 'forecast':
            input_tensor = input_tensor.unsqueeze(1)  # add in batch size of 1

        encoder_hidden = self.encoder.init_hidden(input_tensor.shape[1])
        encoder_hidden = (encoder_hidden[0].to(device), encoder_hidden[1].to(device))
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)


        # Initialize outputs tensor
        outputs = torch.zeros(target_len, input_tensor.shape[1], input_tensor.shape[2])
        outputs = outputs.to(device)

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden

        outputs, decoder_hidden = self.decoder(decoder_input,
                                               decoder_hidden,
                                               outputs=outputs,
                                               target_len=target_len,
                                               prediction_type=prediction_type)

        if prediction_type == 'forecast':
            outputs = outputs.detach()


        return outputs

