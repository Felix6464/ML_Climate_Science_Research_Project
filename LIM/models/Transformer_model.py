from LIM.neural_networks.train_eval_infrastructure import RMSELoss, TimeSeriesDataset, TimeSeriesDatasetnp, TimeSeriesDropout

import torch
import torch.nn as nn
import math
import time
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
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



    def train_model(self, train_dataloader, eval_dataloader, optimizer, config):

        if config["wandb"] is True:
            wandb.init(project=f"SST-{config['model_label']}", config=config, name=config['name'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(device)

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
                    hidden = (hidden[0].to(device), hidden[1].to(device))
                    encoder_output, encoder_hidden = self.encoder(input_batch, hidden)


                    decoder_input = encoder_hidden[0].permute(1, 0, 2)

                    outputs, decoder_hidden = self.decoder(decoder_input,
                                                           hidden,
                                                           outputs,
                                                           config["output_window"])


                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # Backpropagation and weight update
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.7)
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

        for batch, i in enumerate(range(0, len(train_data), batch_size)):  # Now len-1 is not necessary
            # data and target are the same shape with (input_window,batch_len,1)
            data, targets = get_batch(train_data, i, batch_size)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.7)
            optimizer.step()

            total_loss += loss.item()
            log_interval = int(len(train_data) / batch_size / 5)
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.6f} | {:5.2f} ms | '
                      'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                                  elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def plot_and_loss(eval_model, data_source, epoch):
        eval_model.eval()
        total_loss = 0.
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)
        with torch.no_grad():
            # for i in range(0, len(data_source) - 1):
            for i in range(len(data_source)):  # Now len-1 is not necessary
                data, target = get_batch(data_source, i, 1)  # one-step forecast
                output = eval_model(data)
                total_loss += criterion(output, target).item()
                test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
                truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

        # test_result = test_result.cpu().numpy() -> no need to detach stuff..
        len(test_result)

        plt.plot(test_result, color="red")
        plt.plot(truth[:500], color="blue")
        plt.plot(test_result - truth, color="green")
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('graph/transformer-epoch%d.png' % epoch)
        plt.close()
        return total_loss / i

    # predict the next n steps based on the input data
    def predict_future(eval_model, data_source, steps):
        eval_model.eval()
        total_loss = 0.
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)
        data, _ = get_batch(data_source, 0, 1)
        with torch.no_grad():
            for i in range(0, steps):
                output = eval_model(data[-input_window:])
                # (seq-len , batch-size , features-num)
                # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
                data = torch.cat((data, output[-1:]))  # [m,m+1,..., m+n+1]

        data = data.cpu().view(-1)

        # I used this plot to visualize if the model pics up any long therm structure within the data.
        plt.plot(data, color="red")
        plt.plot(data[:input_window], color="blue")
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('graph/transformer-future%d.png' % steps)
        plt.show()
        plt.close()

    def evaluate(eval_model, data_source):
        eval_model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        eval_batch_size = 1000
        with torch.no_grad():
            # for i in range(0, len(data_source) - 1, eval_batch_size): # Now len-1 is not necessary
            for i in range(0, len(data_source), eval_batch_size):
                data, targets = get_batch(data_source, i, eval_batch_size)
                output = eval_model(data)
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
        return total_loss / len(data_source)

    train_data, val_data = get_data()
    model = TransformerModel().to(device)

    criterion = nn.MSELoss()
    lr = 0.005
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 10  # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data)
        if (epoch % 5 == 0):
            val_loss = plot_and_loss(model, val_data, epoch)
            predict_future(model, val_data, 200)
        else:
            val_loss = evaluate(model, val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                    time.time() - epoch_start_time),
                                                                                                      val_loss,
                                                                                                      math.exp(
                                                                                                          val_loss)))
        print('-' * 89)

        # if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    best_model = model

        scheduler.step()