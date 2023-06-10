from transformers import BertTokenizer, BertModel
from setting.settings import model_name, tokenizer_name
from embed.apivocab_embed import get_encoded_api
import torch
import numpy as np
from embed.apivocab_embed import get_api_list_embed



class embed_process:
    def __init__(self, data_list, model=None, tokenizer=None, device=None):
        self.data_list = data_list
        if model is None:
            self.model = BertModel.from_pretrained(model_name)
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {self.device} device")
        self.embed_ids = get_encoded_api(self.tokenizer, self.data_list)
        self.embed_array = get_api_list_embed(self.model, self.embed_ids, self.device)
