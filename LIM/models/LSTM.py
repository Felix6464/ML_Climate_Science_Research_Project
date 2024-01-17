from torch.utils.data import DataLoader, Dataset
from LIM.neural_networks.train_eval_infrastructure import RMSELoss, TimeSeriesDataset, TimeSeriesDatasetnp, TimeSeriesDropout

import numpy as np
import random
from tqdm import trange
import torch
import torch.nn as nn
import wandb



class LSTM_Sequence_Prediction_Base(nn.Module):
    """
    train LSTM encoder-decoder and make predictions
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        '''

        super(LSTM_Sequence_Prediction_Base, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstms = nn.ModuleList()

        for i in range(num_layers):
            input_size = input_size if i == 0 else hidden_size
            self.lstms.append(nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout))

        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input, hidden, outputs=None, training_prediction=None, target_len=None, prediction_type=None):

        input = input.unsqueeze(1)

        for t in range(target_len):
            for i in range(self.num_layers):
                lstm_out, hidden = self.lstms[i](input, hidden)
                input = hidden[0].view(hidden[0].shape[1], 1, hidden[0].shape[2])

            output = self.linear(lstm_out.squeeze(0))
            outputs[:, t, :] = output[:, 0, :]
            input = output


        return outputs, hidden


    def _predict(self, input_tensor, target_len, prediction_type='test'):

            """
            : param input_tensor:      input raw_data (seq_len, input_size); PyTorch tensor
            : param target_len:        number of target values to predict
            : return np_outputs:       np.array containing predicted values; prediction done recursively
            """

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = input_tensor.to(device)

            # encode input_tensor
            if prediction_type == 'forecast':
                input_tensor = input_tensor.unsqueeze(1)  # add in batch size of 1


            # Initialize outputs tensor
            outputs = torch.zeros(input_tensor.shape[0], target_len, input_tensor.shape[2])
            outputs = outputs.to(device)

            # decode input_tensor
            input = input_tensor[:, -1, :]

            hidden = (torch.zeros(self.num_layers, input.shape[0], self.hidden_size),
                      torch.zeros(self.num_layers, input.shape[0], self.hidden_size))
            hidden = (hidden[0].to(device), hidden[1].to(device))


            outputs, decoder_hidden = self.forward(input,
                                                   hidden,
                                                   outputs=outputs,
                                                   target_len=target_len,
                                                   prediction_type=prediction_type)

            if prediction_type == 'forecast':
                outputs = outputs.detach()


            return outputs


    def _forward(self, input_batch, outputs, config, target_batch=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden = (torch.zeros(self.num_layers, config["batch_size"], self.hidden_size, requires_grad=True),
                  torch.zeros(self.num_layers, config["batch_size"], self.hidden_size, requires_grad=True))
        hidden = (hidden[0].to(device), hidden[1].to(device))

        input = input_batch[:, -1, :]

        outputs, decoder_hidden = self.forward(input,
                                               hidden,
                                               outputs,
                                               config["training_prediction"],
                                               config["output_window"])

        return outputs, decoder_hidden