import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from setting import *
from prehandle.Dataset import MyDataset
from tqdm import tqdm


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers, seq_length):
        super(TransformerEncoder, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            # input: (seq_len, batch_size, embedding_dim)
            nn.TransformerEncoderLayer(embedding_dim, nhead=8, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc = nn.Linear(seq_length, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        x = self.transformer_encoder(x)  # x: (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)
        x = x.mean(dim=2)  # (batch_size, seq_len)
        x = self.fc(x)  # (batch_size, hidden_dim)
        x = nn.ReLU()(x)
        x = self.out(x)  # (batch_size, num_classes)
        return x


def train(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch + 1))
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                pbar.set_postfix({'loss': '{:.6f}'.format(loss.item())})
                pbar.update(1)
    return loss_list


def predict(model, data_loader, device, K=3):
    model.eval()
    predictions = []
    tar = []
    with torch.no_grad():
        for (data, target) in tqdm(data_loader):
            data = data.to(device)
            output = model(data)
            topk_values, topk_indices = torch.topk(output, k=K, dim=1)
            predictions.extend(topk_indices.tolist())
            tar.extend(target.tolist())
    return predictions, tar


def evaluate(y_true, y_pred):
    abnormal = []
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in y_pred[i]:
            correct += 1
        else:
            abnormal.append((i, y_true[i], y_pred[i]))
    accuracy = 100 * correct / len(y_true)
    return accuracy, abnormal


# 将间断的不正常数据合并为连续的不正常区域
def get_consistent_abnormal(abnormal):
    consistent_abnormal = []
    left, right = abnormal[0][0], abnormal[0][0] + 1
    for i in range(1, len(abnormal)):
        if abnormal[i][0] == right:
            right += 1
        else:
            consistent_abnormal.append((left, right))
            left, right = abnormal[i][0], abnormal[i][0] + 1
    consistent_abnormal.append((left, right))
    return consistent_abnormal
