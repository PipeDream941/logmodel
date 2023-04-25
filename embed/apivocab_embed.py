# 选择tokenizer和model生成api的embedding的npy文件，并保存api2token字典
# 参数：
#   - tokenizer_name: 选择的tokenizer
#   - model_name: 选择的model
#   - api_csv_path: api_csv文件路径
#   - api_embed_path: 保存api embedding的npy文件路径
#   - gen_dic: 是否生成api2token字典
#   - api_dic_path: 保存api2token字典
# 输出：
#   - api_embed_path: 保存api embedding的npy文件
#   - api_dic_path: 保存api2token字典
import sys

sys.path.extend(['D:\\workspace\\centific\\apimodel'])

from typing import List
import torch
import os
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel
import getopt
import sys
from setting.settings import *
from gen_token_dic import gen_dic


def get_encoded_api(tokenizer, api_list: List[str]) -> list[torch.Tensor]:
    """
    定义tokenizer规则
    :param tokenizer: tokenizer
    :param api_list: List[str]
    :return: api_encoded_list: List
    """
    api_encoded_list = []
    for api in api_list:
        encoded_dict = tokenizer.encode_plus(
            api,  # Sentence to encode.
            add_special_tokens=False,  # Not Add '[CLS]' and '[SEP]'
            return_attention_mask=False,
            return_tensors='pt'
        )
        api_encoded_list.append(encoded_dict['input_ids'])
    return api_encoded_list


def get_opts(argv):
    default_params = {"gen_dic": "False"}
    try:
        opts, args = getopt.getopt(argv, "h", ["help", "gen_dic="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    params = default_params
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("选择tokenizer和model生成api的embedding的npy文件，并保存api2token字典")
            print("参数：")
            print("  - gen_dic: 是否生成api2token字典")
            print("输出：")
            print("  - api_embed_path: 保存api embedding的npy文件")
            print("  - api_dic_path: 保存api2token字典")
            sys.exit()
        elif opt == "--gen_dic":
            params["gen_dic"] = arg
    return params


def main():
    params = get_opts(sys.argv[1:])

    # 选取csv文件中的api, api_list[1:-1] 删去表头和结尾的空格
    with open(api_csv_path, 'r') as f:
        api_list = f.read().split("\n")
    api_list = [api.split(",")[1] for api in api_list[1:-1]]

    # 选取tokenizer和model
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertModel.from_pretrained(model_name)

    api_tokenized_ids = get_encoded_api(tokenizer, api_list)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 如果存在api_embeddings.npy文件，则删除
    if os.path.exists(api_embed_path):
        os.remove(api_embed_path)

    length = len(api_tokenized_ids)
    api_embed_array = np.empty((length, model.config.hidden_size))

    for i in tqdm(range(length)):
        inputs = api_tokenized_ids[i].to(device)
        outputs = model(inputs)
        outputs = outputs[0].mean(dim=1, keepdim=False)  # [n, 768] -> [1, 768]
        outputs = outputs.squeeze(0)  # [1, 768] -> [768]
        api_embed_array[i] = outputs.cpu().detach().numpy()
    # embedding 设置为input_ids的平均值，不设定最大长度
    np.save(api_embed_path, api_embed_array)
    if params["gen_dic"] == "True":
        gen_dic(api_csv_path, api_embed_path, api_dic_path)


if __name__ == '__main__':
    main()
