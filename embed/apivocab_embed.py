# 选择tokenizer和model生成api的embedding的npy文件
# 参数：
#   - tokenizer_name: 选择的tokenizer
#   - model_name: 选择的model
#   - api_csv_path: api_csv文件路径
#   - api_embed_path: 保存api embedding的npy文件路径
#   - gen_dic: 是否生成api_dic字典
#   - api_dic_path: 保存api_dic字典
# 输出：
#   - api_embed_path: 保存api embedding的npy文件
#   - api_dic_path: 保存api_dic字典
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


def get_api_list_embed(model, api_list, device):
    """
    获取api_list的embedding
    :param model: bert model
    :param api_list: List[str]
    :param device:
    :return: List[np.array]
    """
    model = model.to(device)
    length = len(api_list)
    api_embed_array = np.empty((length, model.config.hidden_size))
    for i in tqdm(range(length)):
        inputs = api_list[i].to(device)
        outputs = model(inputs)
        outputs = outputs[0].mean(dim=1, keepdim=False)  # [n, 768] -> [1, 768]
        outputs = outputs.squeeze(0)  # [1, 768] -> [768]
        api_embed_array[i] = outputs.cpu().detach().numpy()
    return api_embed_array





def read_api_csv(_api_csv_path):
    with open(_api_csv_path, 'r') as f:
        api_list = f.read().split("\n")
    api_list = [api.split(",")[1] for api in api_list[1:-1]]
    special_token = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]
    # 加入bert的全部special token
    api_list.extend(special_token)
    return api_list


def main():
    # 读取api列表,加入bert的全部special token
    api_list = read_api_csv(api_csv_path)
    # 选取tokenizer和model
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertModel.from_pretrained(model_name)
    # 获取api列表的tokenized ids
    api_tokenized_ids = get_encoded_api(tokenizer, api_list)
    # 如果存在api_embeddings.npy文件，则删除
    if os.path.exists(api_embed_path):
        os.remove(api_embed_path)
    # 获取api列表的embedding
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    api_embed_array = get_api_list_embed(model, api_tokenized_ids, device)

    np.save(api_embed_path, api_embed_array)


if __name__ == '__main__':
    main()
