import sys

sys.path.extend(['D:\\workspace\\centific\\apimodel'])

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from setting.settings import api_embed_path, api_embed_ae_path
from tqdm import tqdm


class AutoEncoder(nn.Module):
    def __init__(self, _input_dim=768, _hidden_size=None, _output_dim=768):

        super(AutoEncoder, self).__init__()
        self.input_dim = _input_dim
        if _hidden_size is None:
            self.hidden = [512, 256, 128, 64]
        else:
            self.hidden = _hidden_size
        self.output_dim = _output_dim
        self.len_hidden = len(self.hidden)

        layers = [nn.Linear(self.input_dim, self.hidden[0]), nn.ReLU(True)]
        for i in range(self.len_hidden - 1):
            layers.append(nn.Linear(self.hidden[i], self.hidden[i + 1]))
            layers.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*layers)

        layers = []
        for i in range(self.len_hidden - 1, 0, -1):
            layers.append(nn.Linear(self.hidden[i], self.hidden[i - 1]))
            layers.append(nn.ReLU(True))
        layers += [nn.Linear(self.hidden[0], self.output_dim), nn.Tanh()]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(_model, _optimizer, _criterion, _dataloader, _num_epochs):
    epoch_loss_list = []
    with tqdm(total=_num_epochs) as pbar:
        for epoch in range(_num_epochs):
            loss_list = []
            for batch in _dataloader:
                _optimizer.zero_grad()  # 梯度清零
                input_data = batch[0]
                output_data = _model(input_data)
                loss = _criterion(output_data, input_data)
                loss.backward()
                _optimizer.step()
                loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)
            epoch_loss_list.append(loss_mean)
            pbar.set_postfix({'epoch': '{:02d}'.format(epoch + 1), 'loss': '{:.6f}'.format(loss_mean)})
            pbar.update(1)

    return epoch_loss_list
