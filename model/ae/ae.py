import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

api_embed_path = r"D:\workspace\centific\apimodel\data\api_embed_mean.np"
api_embed_ae_path = r"D:\workspace\centific\apimodel\data\api_embed_ae_10.npy"


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
    train_loss = []
    for epoch in range(_num_epochs):
        for batch in _dataloader:
            _optimizer.zero_grad()  # 梯度清零
            input_data = batch[0]
            output_data = _model(input_data)
            loss = _criterion(output_data, input_data)
            loss.backward()
            _optimizer.step()
            train_loss.append(loss.item())
        print(f"Epoch {epoch + 1}/{_num_epochs}, Train Loss: {np.mean(train_loss):.4f}")


# 定义超参数
num_epochs = 50
batch_size = 64
lr = 0.001
input_dim = 768
output_dim = 768
hidden_size = [512, 256, 128, 64, 32, 16, 8]

# 准备数据
data = np.load(api_embed_path)
data = torch.from_numpy(data).float()

# 定义模型、优化器和损失函数
model = AutoEncoder(input_dim, hidden_size, output_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='sum')

# 转移到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# 定义数据加载器
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练模型
train(model, optimizer, criterion, dataloader, num_epochs)

# 保存降维后的数据
encoder = model.encoder
encoded = encoder(data).cpu().detach().numpy()
np.save(api_embed_ae_path, encoded)
