# 参数：
#   - api_embed_path: 保存api embedding的npy文件路径
#   - api_dic_path: 保存api2token字典的路径
# 输出：
#   - api_dic_path: 保存api2token字典
import sys
sys.path.extend(['D:\\workspace\\centific\\apimodel'])
import os
import numpy as np
from setting.settings import api_embed_ae_path, api_dic_path, api_csv_path


def gen_dic(_api_csv_path, _api_embed_path: str, _api_dic_path: str):
    """
    生成api2token字典
    :param _api_csv_path: api_csv文件路径
    :param _api_embed_path: 保存api embedding的npy文件路径
    :param _api_dic_path: 保存api2token字典的路径
    :return: None
    """
    api_embedding = np.load(_api_embed_path, allow_pickle=True)
    api2token = {}

    with open(_api_csv_path, 'r') as f:
        api_list = f.read().split("\n")
    for api in api_list[1:-1]:
        # api_list[1:-1] 删去表头和结尾的空格
        # index从0开始
        index, api_fun = api.split(",")
        api2token[api_fun] = api_embedding[int(index)]
    # 将空api的embedding置为0,长度为api_embedding文件的维度
    api2token[""] = np.zeros(api_embedding.shape[1])
    np.save(_api_dic_path, api2token)
    print("api2token字典已保存至", _api_dic_path)


def main():
    # 如果存在api2token.npy文件，则删除
    if os.path.exists(api_dic_path):
        os.remove(api_dic_path)
    gen_dic(api_csv_path, api_embed_ae_path, api_dic_path)


if __name__ == '__main__':
    main()
