import sys

sys.path.extend(['D:\\workspace\\centific\\apimodel'])

import numpy as np
from tqdm import tqdm
from setting.settings import api_embed_ae_path, api_embed_path


def get_sim_matrix(_api_embed_path: str):
    print("start compute sim matrix, api embed path:", _api_embed_path)
    api_embedding = np.load(_api_embed_path, allow_pickle=True)
    n = len(api_embedding)
    api_embedding = api_embedding.reshape(n, -1)

    # record all api embed size
    api_embed_size = np.zeros(n)
    for i in range(n):
        api_embed_size[i] = round(np.linalg.norm(api_embedding[i]), 2)

    api_sim_matrix = np.zeros((n, n))
    for i in range(n):
        api_sim_matrix[i][i] = 1

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            api_sim_matrix[i][j] = np.sum(api_embedding[i] * api_embedding[j])
            api_sim_matrix[i][j] = round(api_sim_matrix[i][j] / api_embed_size[i] / api_embed_size[j], 2)
            api_sim_matrix[j][i] = api_sim_matrix[i][j]
    return api_sim_matrix


def main():
    api_embed_path1, api_embed_path2 = api_embed_ae_path, api_embed_path
    api_sim_matrix1 = get_sim_matrix(api_embed_path1)
    api_sim_matrix2 = get_sim_matrix(api_embed_path2)
    matrix_sum = np.sum(api_sim_matrix1 * api_sim_matrix2) / np.linalg.norm(api_sim_matrix1) / np.linalg.norm(
        api_sim_matrix2)
    print("降维前后的相似度：", matrix_sum)

if __name__ == '__main__':
    main()
