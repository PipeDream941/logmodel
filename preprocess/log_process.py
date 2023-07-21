import numpy as np
import torch
from pathlib2 import Path

from log_model.transformerdata import get_position_encoding
from prehandle import ContextDataset
from preprocess import ApiLog
from preprocess.oov_process import get_embed_mat


def file_read_process(path: str) -> ApiLog:
    """
    读取单个文件并处理
    :param path: file path(str)
    :return: Class ApiLog
    """
    path = Path(path)
    apilog = ApiLog(path)
    apilog.data_factorize()
    apilog.add_tf()
    return apilog


def files_read_process(path_list: list[str]) -> list[ApiLog]:
    apilog_list = []
    for path in path_list:
        apilog_list.append(file_read_process(path))
    return apilog_list


def log2dataset(apilog: ApiLog, prototype_dict: dict, oov_tensor_dict: dict) -> ContextDataset:
    """
    将apilog转化为dataset
    :param apilog: 单独的apilog
    :param prototype_dict: 所有apilog的prototype字典
    :param oov_tensor_dict: 所有apilog的oov字典
    :return:
    """

    labels = apilog.df["Prototype"].tolist()
    labels_ids = [prototype_dict[i] for i in labels]
    results = apilog.df["Result"].tolist()
    time_encoding = get_position_encoding(apilog.df["Time_Diff"].tolist(), 8)
    embed = get_embed_mat(labels, oov_tensor_dict, time_encoding)
    apilog.df.drop(["merge", "Prototype", "Time_Diff"], axis=1, inplace=True)
    # 每行数据转化为tensor
    feature_tensor = torch.from_numpy(apilog.df.values).float()
    # 拼接数据
    feature_tensor = torch.cat((feature_tensor, embed), dim=1)
    feature_tensor = feature_tensor.to(torch.float)
    labels_ids = torch.tensor(labels_ids, dtype=torch.long)
    results = torch.tensor(results, dtype=torch.long)
    labels_ids = labels_ids.to(torch.long)
    results = results.to(torch.long)
    tars = torch.cat((labels_ids.unsqueeze(1), results.unsqueeze(1)), dim=1)
    dataset = ContextDataset(feature_tensor, tars)
    return dataset


def logs2dataset(logs_list: list[ApiLog], prototype_dict: dict, oov_tensor_dict: dict) -> ContextDataset:
    """
    将logs_list转化为dataset
    :param logs_list: log列表
    :return: dataset
    """
    dataset = log2dataset(logs_list[0], prototype_dict, oov_tensor_dict)
    for i in range(1, len(logs_list)):
        dataset += log2dataset(logs_list[i], prototype_dict, oov_tensor_dict)
    return dataset
