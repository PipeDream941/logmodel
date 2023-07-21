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
            nn.TransformerEncoderLayer(embedding_dim, nhead=5, dim_feedforward=100),
            num_layers=num_layers
        )
        # 类别预测分支
        self.api_classification = nn.Sequential(
            nn.Linear(seq_length, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        # 结果预测分支
        self.result_classification = nn.Sequential(
            nn.Linear(seq_length, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 3)  # -1, 0, 1
        )

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        x = self.transformer_encoder(x)  # x: (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)
        x = x.mean(dim=2)  # (batch_size, seq_len)
        api_x = self.api_classification(x)  # (batch_size, num_classes)
        result_x = self.result_classification(x)  # (batch_size, 3)
        result_x = result_x - 1  # -1, 0, 1
        return api_x, result_x


def train(model, train_loader, optimizer, criterions, weights, device):
    model.train()
    loss_list = []
    api_criterion, res_criterion = criterions["api"], criterions["res"]
    api_weight, res_weight = weights["api"], weights["res"]
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            pbar.update(1)
            data, target = data.to(device), target.to(device)
            target_api, target_result = target[:, 0], target[:, 1]
            optimizer.zero_grad()
            output_api, output_result = model(data)
            # target_result = target_result + 1  # -1, 0, 1 -> 0, 1, 2
            loss_api = api_criterion(output_api, target_api)
            loss_result = res_criterion(output_result, target_result)
            loss = api_weight * loss_api + res_weight * loss_result
            loss.backward()
            optimizer.step()
            loss_list.append((loss.item(), loss_api.item(), loss_result.item()))
            pbar.set_postfix({'loss': loss.item(), 'loss_api': loss_api.item(), 'loss_result': loss_result.item()})
    return loss_list


def predict(model, data_loader, device, K=3):
    model.eval()
    api_predictions = []
    res_predictions = []
    api_tar = []
    res_tar = []
    with torch.no_grad():
        for (data, target) in tqdm(data_loader):
            data = data.to(device)
            api_true, res_true = target[:, 0], target[:, 1]
            output_api, output_res = model(data)
            print(output_api.shape, output_res.shape)
            print(type(output_api), type(output_res))
            topk_values, topk_indices = torch.topk(output_api, k=K, dim=1)
            pre_res = output_res.argmax(dim=1)
            res_predictions.extend(pre_res.tolist())
            api_predictions.extend(topk_indices.tolist())
            api_tar.extend(api_true.tolist())
            res_tar.extend(res_true.tolist())
    return api_predictions, res_predictions, api_tar, res_tar


def evaluate(api_predictions, res_predictions, api_tar, res_tar):
    abnormal = []
    correct = 0
    n = len(api_predictions)
    for i in range(n):
        if api_tar[i] in api_predictions[i] and res_predictions[i] == res_tar[i]:
            correct += 1
        else:
            abnormal.append((i, res_tar[i], res_predictions[i], api_tar[i], api_predictions[i]))
    accuracy = 100 * correct / n
    print("train_accuracy: {:.2f}%".format(accuracy))
    return abnormal


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
