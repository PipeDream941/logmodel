from model.ae.ae import AutoEncoder, train
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch

class dimension_process:
    def __init__(self,data,hidden_size,num_epochs,batch_size,lr):
        self.data = torch.from_numpy(data).float()
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device } device")

    def get_model(self):
        input_dim = output_dim = self.data.shape[1]
        model = AutoEncoder(input_dim, self.hidden_size, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction='sum')
        return model,optimizer,criterion

    def dimension(self):
        model,optimizer,criterion = self.get_model()
        model = model.to(self.device)
        data = self.data.to(self.device)

        # 定义数据加载器
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        loss_list = train(model, optimizer, criterion, dataloader, self.num_epochs)
        return model,loss_list

    def show_loss(self,loss_list):
        import matplotlib.pyplot as plt
        plt.plot(loss_list)
        plt.show()

    def get_encoded(self,model):
        model = model.to(self.device)
        self.data = self.data.to(self.device)
        encoder = model.encoder
        encoded = encoder(self.data).cpu().detach().numpy()
        return encoded




