import numpy as np
import torch



class InputData:
    def __init__(self, df, api2token, model_dim=768, padding_length=50):
        self.df = df
        self.log_length = len(self.df)
        self.model_dim = model_dim
        self.padding_length = padding_length
        self.api2token = api2token
        self.api2index = self.get_api2index()
        self.position_encodings = self.get_position_encoding(self.df["time"].tolist())
        self.input_data = self.get_data()
        self.label = self.get_label()

    def get_position_encoding(self, time_list):
        position_encodings = np.zeros((self.log_length, self.model_dim))
        for pos in range(self.log_length):
            time = time_list[pos]
            for i in range(self.model_dim):
                position_encodings[pos, i] = time / np.power(10000, (i - i % 2) / self.model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])  # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])  # 2i+1
        return position_encodings

    def get_data(self):
        train_data = np.ndarray((self.log_length, self.model_dim))
        for i in range(self.log_length):
            api = self.df["Prototype"].iloc[i]
            if api in self.api2token:
                embed = self.api2token[api]
            else:
                embed = np.zeros(self.model_dim)
            pos_enc = self.position_encodings[i]
            train_data[i] = embed + pos_enc
            if self.df["Result"].iloc[i] == "OK":
                train_data[i] += 1
            else:
                train_data[i] -= 1
        return train_data

    def get_api2index(self):
        api2index = {}
        for index, api in enumerate(self.api2token.keys()):
            api2index[api] = index + 1
        return api2index

    def get_label(self):
        return self.df["Prototype"].tolist()

    def get_train_data(self):
        apis = []
        for api in self.label:
            if api in self.api2index.keys():
                apis.append(self.api2index[api])
            else:
                apis.append(0)
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
