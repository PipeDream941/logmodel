import torch
from torch.utils.data import TensorDataset


class Dataset:
    def __init__(self, data, label, api2index, model_dim=768, padding_length=50):
        self.data = data
        self.label = label
        self.api2index = api2index
        self.padding_length = padding_length
        self.model_dim = model_dim
        self.train_data = self.get_train_data()

    def get_train_data(self):
        apis = []
        for api in self.label:
            if api in self.api2index.keys():
                apis.append(self.api2index[api])
            else:
                apis.append(0)

        apis = torch.tensor(apis)
        train_data = torch.from_numpy(self.data)

        # padding
        padding_data = torch.zeros(self.padding_length, self.model_dim)

        train_data = torch.cat((padding_data, train_data, padding_data), dim=0)

        apis = torch.cat((torch.zeros(self.padding_length), apis, torch.zeros(self.padding_length)), dim=0)

        train_data = TensorDataset(train_data, apis)
        return train_data
