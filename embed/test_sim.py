import numpy as np
from tqdm import tqdm

api_embed_path = "../data/api_embed_ae_10.npy"
sim_matrix_path = "../data/api_sim_matrix_ae_10.npy"

api_embedding = np.load(api_embed_path,allow_pickle=True)

n = len(api_embedding)
api_embedding = api_embedding.reshape(n,-1)

# record all api embed size
api_embed_size = np.zeros(n)
for i in range(n):
    api_embed_size[i] = np.linalg.norm(api_embedding[i])

api_sim_matrix = np.zeros((n,n))
for i in range(n):
    api_sim_matrix[i][i] = 1

for i in tqdm(range(n)):
    for j in range(i+1,n):
        api_sim_matrix[i][j] = np.sum(api_embedding[i] * api_embedding[j])/ api_embed_size[i] / api_embed_size[j]
        api_sim_matrix[j][i] = api_sim_matrix[i][j]

np.save(sim_matrix_path,api_sim_matrix)