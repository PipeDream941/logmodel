from pathlib2 import Path

# 数据集
root_path = Path(r"D:\workspace\centific\Test_Data\2207-Start 11_Data")
repro_path = root_path / "repro"
nonrepro_path = root_path / "nonrepro"

project_path = Path(r"D:\workspace\centific\apimodel")
# 数据路径
data_path = project_path / "data"
api_dic_path = data_path / "api_dic.npy"
api_csv_path = data_path / "apiFunList.csv"
api_embed_path = data_path / "api_embeddings.npy"
api_embed_ae_path = data_path / "api_embed_ae.npy"

model_name = 'bert-base-uncased'
tokenizer_name = 'bert-base-uncased'

