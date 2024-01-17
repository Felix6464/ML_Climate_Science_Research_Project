from LIM.neural_networks.train_eval_infrastructure import RMSELoss, TimeSeriesDataset, TimeSeriesDatasetnp, TimeSeriesDropout
import torch
import torch.nn as nn
import torch.nn.init as init


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


    def _predict(self, input_tensor, target_len, prediction_type='test'):

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


    def _forward(self, input_batch, outputs, config, target_batch=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder forward pass
        hidden = self.encoder.init_hidden(config["batch_size"])
        hidden = hidden.to(device)
        encoder_output, encoder_hidden = self.encoder(input_batch, hidden)

        decoder_input = encoder_hidden.permute(1, 0, 2)

        outputs, decoder_hidden = self.decoder(decoder_input,
                                               hidden,
                                               outputs,
                                               config["output_window"])

        return outputs, decoder_hidden