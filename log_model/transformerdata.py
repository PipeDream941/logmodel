import numpy as np
import torch


class TransformerData:
    def __init__(self, df, api_dic, model_dim=768, padding_length=50):
        self.df = df
        self.log_length = len(self.df)
        self.model_dim, self.padding_length = model_dim, padding_length
        self.api_dic = api_dic
        self.labels = self.get_label()
        self.api2index, self.index2api = self.get_api2index()
        self.log_api_remove_outlier() # remove outlier
        self.position_encodings = self.get_position_encoding(self.df["time"].tolist())
        self.input_data = self.get_trans_data()


    def get_position_encoding(self, time_list: list) -> np.ndarray:
        """
        add time series information into data using position encoding
        :param time_list:
        :return: position_encodings
        """
        position_encodings = np.zeros((self.log_length, self.model_dim))
        for pos in range(self.log_length):
            time = time_list[pos]
            for i in range(self.model_dim):
                position_encodings[pos, i] = time / np.power(10000, (i - i % 2) / self.model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])  # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])  # 2i+1
        return position_encodings

    def log_api_remove_outlier(self):
        """
        sometimes the log data prototype is not in api_dic, so we need to replace them with ""
        :return: None
        """
        for i in range(self.log_length):
            self.df["Prototype"].iloc[i].split(" ")
            api = self.df["Prototype"].iloc[i]
            if api not in self.api_dic.keys():
                self.df["Prototype"].iloc[i] = ""
        return

    def get_trans_data(self):
        """
        get data for transformer model, add position encoding and result feature
        :return:
        """
        data = np.ndarray((self.log_length, self.model_dim))  # 先构造一个(log_length, model_dim)的全0矩阵
        for i in range(self.log_length):
            api = self.df["Prototype"].iloc[i]

            _,embed = self.api_dic[api][1]

            pos_enc = self.position_encodings[i]
            # 添加Position Encoding
            data[i] = embed + pos_enc
            # 添加tfidf特征
            data[i] = np.concatenate((data[i], np.array(self.df["tfidf"].iloc[i])))
            # 添加Result特征
            if self.df["Result"].iloc[i] == "OK":
                data[i] = np.concatenate((np.array([1]), data[i]))
            else:
                data[i] = np.concatenate((np.array([0]), data[i]))
        return data

    def get_api2index(self):
        api2index = {}
        index2api = {}
        for index, api in enumerate(set(self.labels)):
            api2index[api] = index + 1
            index2api[index + 1] = api
        return api2index, index2api

    def get_label(self):
        return self.df["Prototype"].tolist()

    def get_train_data(self):

        apis = [self.api2index[api] for api in self.labels]
        api_set = set(apis)
        apis = torch.tensor(apis)
        train_data = torch.from_numpy(self.input_data)

        # padding
        padding_data = torch.zeros(self.padding_length, self.model_dim)

        train_data = torch.cat((padding_data, train_data, padding_data), dim=0)

        apis = torch.cat((torch.zeros(self.padding_length), apis, torch.zeros(self.padding_length)), dim=0)

        train_data = train_data.float()
        apis = apis.long()

        return train_data, apis, api_set



