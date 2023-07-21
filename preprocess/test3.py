import pickle


from torch import nn
from model.transformer.trans_model import TransformerEncoder
from torch.utils.data import DataLoader
from torch import optim
from preprocess.utils import *
import random
from setting import config
from pathlib2 import Path
from preprocess.log_process import files_read_process
from preprocess.oov_process import get_oov_embed, get_prototype_dict

print("Step1： 文件读取")
# train_files_path = r"D:\workspace\centific\Test_Data\BugLogList\CaptureWiz\norepro"
# test_files_path = r"D:\workspace\centific\Test_Data\BugLogList\CaptureWiz\repro"
train_files_path = r"D:\workspace\centific\Test_Data\buglog_from_frank\norepro"
test_files_path = r"D:\workspace\centific\Test_Data\buglog_from_frank\repro"
train_csvs_path = list(Path(train_files_path).glob("*.csv"))
test_csvs_path = list(Path(test_files_path).glob("*.csv"))
train_apilogs_list = files_read_process(train_csvs_path)
test_apilogs_list = files_read_process(test_csvs_path)

print("Step2：OOV Process")
logs_list = train_apilogs_list + test_apilogs_list
prototype, prototype_dict = get_prototype_dict(apilog_list=logs_list)
oov_tensor_dict = get_oov_embed(_prototype=prototype)

print("Step3: 处理数据")
from preprocess.log_process import logs2dataset

train_dataset = logs2dataset(train_apilogs_list, prototype_dict, oov_tensor_dict)
test_dataset = logs2dataset(test_apilogs_list, prototype_dict, oov_tensor_dict)

# print("加载数据")
# train_dataset = pickle.load(open('train_dataset.pkl', 'rb'))
# test_dataset = pickle.load(open(r"D:\workspace\centific\apimodel\preprocess\test_dataset.pkl", 'rb'))

print("Step4: dataloader，模型，优化器，损失函数")
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
train_loader_not_shuffle = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
model = TransformerEncoder(config.embedding_dim, config.hidden_dim, config.num_classes, config.num_layers,
                           config.seq_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
crtierions = {"api": nn.CrossEntropyLoss(), "res": res_criteria}
weights = {"api": 1, "res": 0}
optimizer = optim.Adam(model.parameters(), lr=config.lr)

print("Step5: 模型训练")
trainer(model, config.epochs, train_loader, optimizer, crtierions, weights, device)

print("Step6: test_loader预测结果")
test_api_pres, test_res_pres, test_api_tars, test_res_tars = predict(model, test_loader, device, k=3)
test_api_ab, test_res_ab, test_res_ab_fail_cons = get_anomaly(test_api_pres, test_res_pres, test_api_tars,
                                                              test_res_tars)
