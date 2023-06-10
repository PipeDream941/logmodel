import numpy as np

from preprocess import ApiLog
from setting.settings import api_dic_path
from log_model import TransformerData
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


def get_special_token(token_name, model_dim=15, padding_length=50):
    api_embed_dic = np.load(r"D:\workspace\centific\apimodel\data\api_dic.npy", allow_pickle=True).item()
    _, token = api_embed_dic[token_name]
    padding_tensor = np.array([0] * (model_dim - len(token)))
    token = np.concatenate((token, padding_tensor))
    token_matrix = np.array([token] * padding_length)
    token_matrix = torch.from_numpy(token_matrix).float()
    return token_matrix


class ContextDataset(Dataset):
    def __init__(self, data, tar, padding_length=50, window_size=100, stride=1):
        self.data = data
        self.tar = tar
        self.model_dim = data.shape[1]
        self.padding_length = padding_length
        self.window_size = window_size
        self.stride = stride
        self.padding()
        self.mask = get_special_token("[MASK]", self.model_dim, 1)

    def padding(self):
        start_matrix = get_special_token("[CLS]", self.model_dim, self.padding_length)
        end_matrix = get_special_token("[SEP]", self.model_dim, self.padding_length)

        padding_tar_tensor = np.array([0] * 2)
        padding_tar_matrix = np.array([padding_tar_tensor] * self.padding_length)
        padding_tar_matrix = torch.from_numpy(padding_tar_matrix).long()
        #padding_tar_matrix.shape = (padding_length, 2)
        self.data = torch.cat((start_matrix, self.data, end_matrix), dim=0)
        self.tar = torch.cat((padding_tar_matrix, self.tar, padding_tar_matrix), dim=0)

    def __getitem__(self, index):
        assert index < len(self)
        return torch.cat((self.data[index: index + self.padding_length],
                          self.mask,
                          self.data[index + self.padding_length + 1: index + 2 * self.padding_length + 1]), dim=0), \
            self.tar[index + self.padding_length]

    def __len__(self):
        # len(self.data) - 2 * self.padding_length 为去掉padding后的长度，即原始数据长度
        # 卷积后的数据长度为 H = (L - W + 2P) / S + 1
        return (len(self.data) - self.window_size) // self.stride

    def __add__(self, other):
        new_data = torch.cat((self.data, other.data), dim=0)
        new_tar = torch.cat((self.tar, other.tar), dim=0)
        return ContextDataset(new_data, new_tar, self.padding_length, self.window_size, self.stride)


class MyDataset:
    def __init__(self, path, padding_length=50, stride=1):
        self.path = path
        self.padding_length = padding_length
        self.stride = stride
        self.api_dic = np.load(api_dic_path, allow_pickle=True).item()
        self.model_dim = self.get_model_dim()
        self.vocab_size = len(self.api_dic) + 1
        self.file_list = list(self.path.glob("*.csv"))
        self.file_num = len(self.file_list)
        self.data = self.get_dataset()

    def get_model_dim(self):
        for k, v in self.api_dic.items():
            return len(v[1])

    def get_dataset(self):
        vocab_set = set()
        datalist = []
        for file in tqdm(self.file_list):
            content = ApiLog(file)
            input_data = TransformerData(content.df, self.api_dic, self.model_dim)
            data, label, token_set = input_data.get_train_data()
            vocab_set = vocab_set | token_set  # 合并两个集合
            datalist.append(
                ContextDataset(data, label, padding_length=self.padding_length, window_size=2 * self.padding_length + 1,
                               stride=self.stride))
        combined_dataset = datalist[0]
        for ds in datalist[1:]:
            combined_dataset += ds
        self.vocab_size = len(vocab_set)
        return combined_dataset


if __name__ == "__main__":
    from pathlib2 import Path

    data_path = r"C:\Users\sivan\Downloads\LogdiffTest\difference data\runtimeapi2"
    data_path = Path(data_path)

    data_set = MyDataset(data_path, padding_length=50, stride=1)
