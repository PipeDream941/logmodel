# 项目结构
  - **data/**
    - api2token.npy # api的embedding
    - apiFunList.csv # 所有api的列表
    - api_embedings.csv # api的embedding
  - **embed/**
    - apivocab_embed.py
    - test_sim.ipynb
    - test_sim.py
  - **log/**

    - apilog.py
    - dataset.py
    - inputdata.py
    - log.py
  - **model/**
    - **ae/**
      - ae.ipynb
      - ae.py
    - **transformer/**
      - 1.ipynb
      - dataset.pkl
      - transformer.py
  - **prehandle/**
    - Dataset.py
    - prehandle.ipynb
  - **setting/**
    - hyper.py
    - setting.py
# 运行步骤
注：以下命令行都需要 cd 到项目根目录下 

## Step 1 api词嵌入处理

- api词嵌入处理
`python .\embed\apivocab_embed.py`  
注：需要直接生成字典可以运行`python .\embed\apivocab_embed.py --gen_vocab=True`
- api词嵌入降维 
`python .\embed\dimension.py`
- 保存api词嵌入到api2token.npy
`python .\embed\gen_token_dic.py`