import numpy as np
from setting.settings import api_dic_path
from log import ApiLog, InputData
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


class ContextDataset(Dataset):
    def __init__(self, data, tar, padding_length=50, window_size=100, stride=1):
        self.data = data
        self.tar = tar
        self.padding_length = padding_length
        self.window_size = window_size
        self.stride = stride

    def __getitem__(self, index):
        assert index < len(self)
        return torch.cat((self.data[index: index + self.padding_length],
                         self.data[index + self.padding_length: index + 2 * self.padding_length]),dim=0), \
            self.tar[index + self.padding_length]

    def __len__(self):
        # len(self.data) - 2 * self.padding_length 为去掉padding后的长度，即原始数据长度
        # 卷积后的数据长度为 H = (L - W + 2P) / S + 1
        return (len(self.data) - self.window_size) // self.stride + 1

    def __add__(self, other):

        new_data = torch.cat((self.data, other.data), dim=0)
        new_tar = torch.cat((self.tar, other.tar), dim=0)
        return ContextDataset(new_data, new_tar, self.padding_length, self.window_size, self.stride)


class MyDataset:
    def __init__(self, path, padding_length=50, stride=1):
        self.path = path
        self.padding_length = padding_length
        self.stride = stride
        self.api2token = np.load(api_dic_path, allow_pickle=True).item()
        self.model_dim = self.get_model_dim()
        self.vocab_size = len(self.api2token) + 1
        self.file_list = list(self.path.glob("*.csv"))
        self.file_num = len(self.file_list)
        self.data = self.get_dataset()


    def get_model_dim(self):
        for k, v in self.api2token.items():
            return len(v)



    def get_dataset(self):
        vocab_set = set()
        datalist = []
        for file in tqdm(self.file_list):
            content = ApiLog(file)
            input_data = InputData(content.df, self.api2token, self.model_dim)
            data, label, token_set = input_data.get_train_data()
            vocab_set = vocab_set | token_set # 合并两个集合
            datalist.append(
                ContextDataset(data, label, padding_length=self.padding_length, window_size=2 * self.padding_length + 1,
                               stride=self.stride))
        combined_dataset = datalist[0]
        for ds in datalist[1:]:
            combined_dataset += ds
        # self.vocab_size = len(vocab_set)
        return combined_dataset




if __name__ == "__main__":
    from setting.settings import nonrepro_path

    data_set = MyDataset(nonrepro_path, padding_length=50, stride=1)
