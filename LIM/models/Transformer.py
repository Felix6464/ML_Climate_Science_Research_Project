from LIM.neural_networks.train_eval_infrastructure import RMSELoss, TimeSeriesDataset, TimeSeriesDatasetnp, TimeSeriesDropout

import torch
import torch.nn as nn
import math
import time
import numpy as np
from matplotlib import pyplot as plt
import wandb
from tqdm import trange



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E)
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)

    def _predictX(self, data_source, target_length, config):
        self.eval()
        total_loss = 0.
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)
        with torch.no_grad():
            for i in range(0, target_length):
                output = self.forward(data[-config["input_window"]:])
                # (seq-len , batch-size , features-num)
                # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
                data = torch.cat((data, output[-1:]))  # [m,m+1,..., m+n+1]

        data = data.cpu().view(-1)

        # I used this plot to visualize if the model pics up any long therm structure within the data.
        plt.plot(data, color="red")
        plt.plot(data[:config["input_window"]], color="blue")
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('../results/transformer-future%d.png' % target_length)
        plt.show()
        plt.close()

    def _predict(self, input_tensor, config, prediction_type='test'):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        # Initialize outputs tensor
        outputs = torch.zeros(input_tensor.shape[0], config["output_window"], input_tensor.shape[2])
        outputs = outputs.to(device)

        with torch.no_grad():
            for i in range(0, config["output_window"]):
                output = self.forward(seq[-config["output_window"]:])
                seq = torch.cat((seq, output[-1:]))

        seq = seq.cpu().view(-1).numpy()

        return seq

    def _forward(self, input_tensor, outputs, config, target_batch=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        for i in range(0, config["output_window"]):
            output = self.forward(seq[-config["output_window"]:])
            seq = torch.cat((seq, output[-1:]))

        seq = seq.cpu().view(-1).numpy()

        return seq