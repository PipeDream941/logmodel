import sys

sys.path.extend(['D:\\workspace\\centific\\apimodel'])
import os
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model.ae.ae import AutoEncoder, train
from setting.settings import api_embed_path, api_embed_ae_path

def main():
    # 准备数据
    data = np.load(api_embed_path)
    data = torch.from_numpy(data).float()

    # 定义超参数
    num_epochs = 500
    batch_size = 64
    lr = 0.001
    hidden_size = [512, 256, 128, 64, 32, 16, 8]
    input_dim = output_dim = data.shape[1]

    # 定义模型、优化器和损失函数
    model = AutoEncoder(input_dim, hidden_size, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    # 转移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = model.to(device)
    data = data.to(device)

    # 定义数据加载器
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    loss_list = train(model, optimizer, criterion, dataloader, num_epochs)

    # 展示loss曲线
    plt.plot(loss_list)
    plt.show()

    # 保存降维后的数据
    encoder = model.encoder
    encoded = encoder(data).cpu().detach().numpy()

    # 如果存在api_embeddings.npy文件，则删除
    if os.path.exists(api_embed_ae_path):
        os.remove(api_embed_ae_path)
    # 保存降维后的数据
    np.save(api_embed_ae_path, encoded)


if __name__ == '__main__':
    main()