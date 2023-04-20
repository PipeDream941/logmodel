import os
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

apicsv_path = "apiFunList.csv"
vocab_path = 'data/api_embed_all.np'


api2index = {}
index2api = {}
with open(apicsv_path, 'r') as f:
    api_list = f.read().split("\n")
for api in api_list[1:-1]:
    index, apifun = api.split(",")
    api2index[apifun] = index
    index2api[index] = apifun
api_list = index2api.values()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_encoded_api(api, max_length=26):
    encoded_dict = tokenizer.encode_plus(
        api,    # Sentence to encode.
        add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
        return_attention_mask=False,   # Construct attn. masks.
        max_length=max_length,           # Pad & truncate all sentences.
        padding='max_length',
        return_tensors='pt',     # return numpy tensors.
        truncation=True
    )
    return encoded_dict['input_ids']


api_tokenized_ids = [get_encoded_api(api) for api in api_list]
api_tokenized_ids = torch.cat(api_tokenized_ids, 0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
api_tokenized_ids = api_tokenized_ids.to(device)

batch_size = 32
api_tmber_path = 'data/api_embed_all.np'
# 如果文件存在，就删除
if os.path.exists(api_tmber_path):
    os.remove(api_tmber_path)
length = len(api_tokenized_ids)
api_embed_array = np.empty((length, 26, 768))
for i in range(0, len(api_tokenized_ids), batch_size):
    test = api_tokenized_ids[i:i+batch_size]
    outputs = model(test)
    api_embeddings = outputs[0]
    api_embed_array[i:i+batch_size] = api_embeddings.cpu().detach().numpy()

with open(vocab_path, 'wb') as f:
    np.save(f, api_embed_array)
