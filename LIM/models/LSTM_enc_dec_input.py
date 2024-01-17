from torch.utils.data import DataLoader, Dataset
from LIM.neural_networks.train_eval_infrastructure import RMSELoss, TimeSeriesDataset, TimeSeriesDatasetnp, TimeSeriesDropout

import numpy as np
import random
from tqdm import trange
import torch
import torch.nn as nn
import wandb





class LSTM_Encoder(nn.Module):
    """
    Encodes time-series sequence
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        """
        : param input_size:     the number of features in the input_data
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        """

        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = TimeSeriesDropout(dropout_prob)

        self.lstms = nn.ModuleList()

        for i in range(num_layers):
            input_size = input_size if i == 0 else hidden_size
            self.lstms.append(nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout_prob))


    def forward(self, x_input, encoder_hidden, prediction_type=None, dropout=False):
        """
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        """

        if self.dropout.dropout_prob > 0 and prediction_type != "test":
            dropout = True
            encoder_hidden = self.dropout(encoder_hidden)

        for i in range(self.num_layers):
            lstm_out, hidden = self.lstms[i](x_input, encoder_hidden)
            x_input, encoder_hidden = lstm_out, hidden
        return lstm_out, hidden

    def init_hidden(self, batch_size):
        """
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        """

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class LSTM_Decoder(nn.Module):
    """
    Decodes hidden state output by encoder
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        """
        : param input_size:     the number of features in the input_data
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        """

        super(LSTM_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = TimeSeriesDropout(dropout_prob)

        self.lstms = nn.ModuleList()

        for i in range(self.num_layers):
            input_size = input_size if i == 0 else hidden_size
            self.lstms.append(nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout_prob))

        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, decoder_input, decoder_hidden, outputs=None, target_batch=None, training_prediction=None, target_len=None, teacher_forcing_ratio=None, prediction_type=None):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence
        '''

        decoder_input = decoder_input.unsqueeze(1)


        if training_prediction == 'teacher_forcing':
            # Use teacher forcing
            if random.random() < teacher_forcing_ratio:
                for t in range(target_len):
                    for i in range(self.num_layers):
                        lstm_out, decoder_hidden = self.lstms[i](decoder_input, decoder_hidden)
                        decoder_input = decoder_hidden[0].permute(1, 0, 2)

                    decoder_output = self.linear(lstm_out.squeeze(0))
                    outputs[:, t, :] = decoder_output[:, 0, :]
                    decoder_input = target_batch[:, t, :]

            # Predict recursively
            else:
                for t in range(target_len):
                    for i in range(self.num_layers):
                        lstm_out, decoder_hidden = self.lstms[i](decoder_input, decoder_hidden)
                        decoder_input = decoder_hidden[0].view(decoder_hidden[0].shape[1], 1, decoder_hidden[0].shape[2])

                    decoder_output = self.linear(lstm_out.squeeze(0))
                    outputs[:, t, :] = decoder_output
                    decoder_input = decoder_output

        elif training_prediction == 'mixed_teacher_forcing':
            # Predict using mixed teacher forcing
            for t in range(target_len):
                for i in range(self.num_layers):
                    lstm_out, decoder_hidden = self.lstms[i](decoder_input, decoder_hidden)
                    decoder_output = self.linear(lstm_out.squeeze(0))
                    decoder_input = decoder_hidden[0].permute(1, 0, 2)


                outputs[:, t, :] = decoder_output[:, 0, :]

                # Predict with teacher forcing
                if random.random() < teacher_forcing_ratio:
                    decoder_input = target_batch[:, t, :].unsqueeze(1)

                # Predict recursively
                else:
                    decoder_input = decoder_output

        else:
            # Predict recursively
            for t in range(target_len):
                for i in range(self.num_layers):
                    lstm_out, decoder_hidden = self.lstms[i](decoder_input, decoder_hidden)
                    decoder_output = self.linear(lstm_out.squeeze(0))
                    decoder_input = decoder_hidden[0].permute(1, 0, 2)
                #print(decoder_output.shape)
                outputs[:, t, :] = decoder_output[:, 0, :]
                decoder_input = decoder_output


        return outputs, decoder_hidden


class LSTM_Sequence_Prediction_Input(nn.Module):
    """
    train LSTM encoder-decoder and make predictions
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        '''

        super(LSTM_Sequence_Prediction_Input, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = LSTM_Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout_prob=dropout)
        self.decoder = LSTM_Decoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout_prob=dropout)

    def _predict(self, input_tensor, target_len, prediction_type='test'):

        """
        : param input_tensor:      input raw_data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        if prediction_type == 'forecast':
            input_tensor = input_tensor.unsqueeze(0)

        hidden = self.encoder.init_hidden(input_tensor.shape[0])
        hidden = (hidden[0].to(device), hidden[1].to(device))
        encoder_output, encoder_hidden = self.encoder(input_tensor, hidden, prediction_type)

        # Initialize outputs tensor
        outputs = torch.zeros(input_tensor.shape[0], target_len, input_tensor.shape[2])
        outputs = outputs.to(device)

        # decode input_tensor
        decoder_input = input_tensor[:, -1, :]

        outputs, decoder_hidden = self.decoder(decoder_input,
                                               encoder_hidden,
                                               outputs=outputs,
                                               target_len=target_len,
                                               prediction_type=prediction_type)

        if prediction_type == 'forecast':
            outputs = outputs.detach()

        return outputs


    def _forward(self, input_batch, target_batch, outputs, config):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder forward pass
        hidden = self.encoder.init_hidden(config["batch_size"])
        hidden = (hidden[0].to(device), hidden[1].to(device))
        encoder_output, encoder_hidden = self.encoder(input_batch, hidden)

        # Decoder input for the current batch
        decoder_input = input_batch[:, -1, :]

        outputs, decoder_hidden = self.decoder(decoder_input,
                                               encoder_hidden,
                                               outputs=outputs,
                                               training_prediction=config["training_prediction"],
                                               target_len=config["output_window"],
                                               teacher_forcing_ratio=config["teacher_forcing_ratio"],
                                               target_batch=target_batch)

        return outputs, decoder_hidden

