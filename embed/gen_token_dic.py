# 参数：
#   - api_embed_path: 保存api embedding的npy文件路径
#   - api_dic_path: 保存api_dic字典的路径
# 输出：
#   - api_dic_path: 保存api_dic字典
import sys
sys.path.extend(['D:\\workspace\\centific\\apimodel'])
import os
import numpy as np
from setting.settings import api_embed_ae_path, api_dic_path, api_csv_path
from embed.apivocab_embed import read_api_csv


def gen_dic(_api_csv_path, _api_embed_path: str, _api_dic_path: str):
    """
    生成api_dic字典
    :param _api_csv_path: api_csv文件路径
    :param _api_embed_path: 保存api embedding的npy文件路径
    :param _api_dic_path: 保存api_dic字典的路径
    :return: None
    """
    api_embedding = np.load(_api_embed_path, allow_pickle=True)
    api_dic = {}


    api_list = read_api_csv(_api_csv_path)
    for index, api in enumerate(api_list):
        api_dic[api] = (index, api_embedding[index])
    # 将空api的embedding置为0,长度为api_embedding文件的维度
    api_dic[""] = (len(api_dic.keys()) + 1, np.zeros(api_embedding.shape[1]))
    np.save(_api_dic_path, api_dic)
    print("api_dic字典已保存至", _api_dic_path)


def main():
    # 如果存在api_dic.npy文件，则删除
    if os.path.exists(api_dic_path):
        os.remove(api_dic_path)
    gen_dic(api_csv_path, api_embed_ae_path, api_dic_path)


if __name__ == '__main__':
    main()
