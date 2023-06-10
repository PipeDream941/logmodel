import sys
sys.path.extend(['D:\\workspace\\centific\\apimodel'])

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch


from setting.settings import api_csv_path

def get_prototype_dict(apilog_list: list):
    """
    get the prototype dict from the apilog_list
    :param apilog_list:
    :return:
    """
    prototype = []
    for apilog in apilog_list:
        prototype.extend(apilog.df.Prototype.tolist())
    prototype_set = list(set(prototype))
    prototype_ids = list(range(len(prototype_set)))
    prototype_dict = dict(zip(prototype_set, prototype_ids))
    return prototype, prototype_dict


def get_out_api_dic(prototype: list, api_list: list):
    prototype_cnt = Counter(prototype)
    # out_cnt用于记录转换后的api的出现次数
    out_cnt = Counter()
    # out_api_dict用于记录转换后的api和原始api的对应关系
    out_api_dict = defaultdict(list)

    for k, v in prototype_cnt.items():
        if k not in api_list:
            out_cnt[str(k).lower()] = v
            out_api_dict[str(k).lower()] = [k]
    return out_cnt, out_api_dict


def out_api_dict_merge(out_api_dict: dict, out_cnt: dict):
    for k in list(out_cnt.keys()):
        # 将非字母和数字的字符替换成空格
        tmp = re.sub(r"[^a-z0-9]+", " ", k)
        # 如果是连续的空格，则替换成一个空格
        tmp = re.sub(r"\s+", " ", tmp)
        if tmp != k:
            out_api_dict[tmp] += out_api_dict.pop(k)
            out_cnt[tmp] = out_cnt.pop(k)
    # 把k为空的项转换成unknown
    if "" in out_cnt:
        out_cnt["unknown"] += out_cnt.pop("")
        out_api_dict["unknown"] += out_api_dict.pop("")
    if " " in out_cnt:
        out_cnt["unknown"] += out_cnt.pop(" ")
        out_api_dict["unknown"] += out_api_dict.pop(" ")
    return out_cnt, out_api_dict


def get_oov_embed(_prototype: list):
    # 读取apiFunction，把不属于apiFunction的函数展示出来
    api_function = pd.read_csv(r"D:\workspace\centific\apimodel\data\apiFunList.csv")
    api_name_list = api_function["ApiFunName"].to_list()
    out_cnt, out_api_dict = get_out_api_dic(_prototype, api_name_list)
    out_cnt, out_api_dict = out_api_dict_merge(out_api_dict, out_cnt)
    out_list = sorted(out_cnt.items(), key=lambda x: x[1], reverse=True)
    api_list = list(zip(*out_list))[0]
    # 得到原始api和转换后的api的对应关系
    api_dict = {i: k for k, v in out_api_dict.items() for i in v}
    print("预训练模型词嵌入")
    from embed.embed_process import embed_process

    embed = embed_process(api_list)
    api_embed_array = embed.embed_array
    print("oov词嵌入降维")
    from embed.dimension_process import dimension_process

    dimension = dimension_process(data=api_embed_array,
                                  hidden_size=[512, 256, 128, 64, 32, 16, 8],
                                  num_epochs=1000,
                                  batch_size=64,
                                  lr=0.001)

    ae, dimension_loss_list = dimension.dimension()
    dimension.show_loss(dimension_loss_list)
    encoded = dimension.get_encoded(ae)

    oov_tensor_dict = {}
    for k, v in api_dict.items():
        # k是原始api，v是转换后的api
        index = out_list.index((v, out_cnt[v]))
        oov_tensor_dict[k.lower()] = encoded[index]
    return oov_tensor_dict


def get_embed_mat(proto_list, oov_tensor_dict, pe):
    api_embed_dic = np.load(r"D:\workspace\centific\apimodel\data\api_dic.npy", allow_pickle=True).item()
    embed_mat = []
    for i in range(len(proto_list)):
        api = proto_list[i]
        if api in api_embed_dic:
            embed = api_embed_dic[api][1]
        else:
            embed = oov_tensor_dict[api.lower()]
        embed += pe[i]
        embed_mat.append(embed)
    embed_mat = np.array(embed_mat, dtype=float)
    embed_mat = torch.from_numpy(embed_mat)
    return embed_mat
