import torch

from matplotlib import pyplot as plt
from tqdm import tqdm
from model.transformer.trans_model import train
from typing import List, Tuple


def predict(model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            device: torch.device,
            k: int = 3) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    预测函数， 返回api预测结果， res预测结果， api真实结果， res真实结果
    真实api在在topk中， 则认为预测正确
    :param model: Transformer Encoder
    :param data_loader: DataLoader
    :param device: cuda or cpu
    :param k: top k, default 3
    :return: api_predictions, res_predictions, api_targets, res_targets
    """
    model.eval()
    api_predictions, res_predictions = [], []
    api_targets, res_targets = [], []
    with torch.no_grad():
        for (data, target) in tqdm(data_loader):
            data = data.to(device)
            api_true, res_true = target[:, 0], target[:, 1]
            output_api, output_res = model(data)
            topk_values, topk_indices = torch.topk(output_api, k=k, dim=1)
            pre_res = output_res.argmax(dim=1)
            res_predictions.extend(pre_res.tolist())
            api_predictions.extend(topk_indices.tolist())
            api_targets.extend(api_true.tolist())
            res_targets.extend(res_true.tolist())
    res_predictions = [i - 1 for i in res_predictions]
    return api_predictions, res_predictions, api_targets, res_targets


def only_evaluate(api_predictions, res_predictions, api_tar, res_tar):
    api_correct, res_correct = 0, 0
    fail_num, ok_num, star_num = 0, 0, 0
    n = len(api_predictions)
    for i in range(n):
        if api_tar[i] in api_predictions[i]:
            api_correct += 1
        if res_predictions[i] == res_tar[i]:
            res_correct += 1
            if res_tar[i] == 1:
                ok_num += 1
            elif res_tar[i] == -1:
                fail_num += 1
            else:
                star_num += 1
    pre_ok_num = sum([1 for i in range(n) if res_predictions[i] == 1]) + 1
    pre_fail_num = sum([1 for i in range(n) if res_predictions[i] == -1]) + 1
    pre_star_num = sum([1 for i in range(n) if res_predictions[i] == 0]) + 1

    real_ok_num = sum([1 for i in range(n) if res_tar[i] == 1]) + 1
    real_fail_num = sum([1 for i in range(n) if res_tar[i] == -1]) + 1
    real_star_num = sum([1 for i in range(n) if res_tar[i] == 0]) + 1
    print("ok_precision:{}, fail_precision:{}, star_precision:{}".format(ok_num / pre_ok_num, fail_num / pre_fail_num,
                                                                         star_num / pre_star_num))
    print("ok_recall:{}, fail_recall:{}, star_recall:{}".format(ok_num / real_ok_num, fail_num / real_fail_num,
                                                                star_num / real_star_num))
    print("pre_ok_num:{}, pre_fail_num:{}, pre_star_num:{}".format(pre_ok_num, pre_fail_num, pre_star_num))
    print("real_ok_num:{}, real_fail_num:{}, real_star_num:{}".format(real_ok_num, real_fail_num, real_star_num))
    api_accuracy, res_accuracy = round(100 * api_correct / n, 4), round(100 * res_correct / n, 4)
    print("api_accuracy:{}, res_accuracy:{}".format(api_accuracy, res_accuracy))


def res_criteria(predictions, tar):
    # 统计预测结果中的-1的f1值作为res的loss
    epision = 1e-7
    pre_fail_index = torch.where(predictions == 0)[0]
    tar_fail_index = torch.where(tar == 0)[0]
    tp = len(set(pre_fail_index) & set(tar_fail_index))
    fp = len(pre_fail_index) - tp
    fn = len(tar_fail_index) - tp
    precision = tp / (tp + fp + epision)
    recall = tp / (tp + fn + epision)
    f1 = 2 * precision * recall / (precision + recall + epision)
    return torch.tensor(1 - f1)


def get_continuous_data(data: List[int], gap=2):
    """
    间隔为2的数据认为是连续数据
    :param gap: gap内的数据认为是连续的
    :param data: 原始数据
    :return: List[(start, end)]
    """
    if len(data) < 2:
        return data
    continuous_data = []
    left, right = 0, 1
    while right < len(data):
        if data[right] - data[right - 1] > gap:
            continuous_data.append((data[left], data[right - 1]))
            left = right
        right += 1
    continuous_data.append((data[left], data[right - 1]))
    return continuous_data


def trainer(model, epochs, loader, optimizer, crtierions, weights, device):
    for i in range(epochs):
        print("epoch:{}".format(i + 1))
        train_loss_list = train(model, loader, optimizer, crtierions, weights, device)
        # loss_list的元素分别是all_loss, api_loss, res_loss
        all_loss_list, api_loss_list, res_loss_list = zip(*train_loss_list)
        plt.plot(all_loss_list)
        plt.plot(api_loss_list)
        plt.plot(res_loss_list)
        plt.show()
        api_predictions, res_predictions, api_tar, res_tar = predict(model, loader, device)
        only_evaluate(api_predictions, res_predictions, api_tar, res_tar)


def get_anomaly(api_predictions, res_predictions, api_targets, res_targets):
    api_abnormal, result_abnormal, only_fail_abnormal = [], [], []
    api_correct, res_correct = 0, 0
    n = len(api_predictions)
    for i in range(n):
        if api_targets[i] in api_predictions[i]:
            api_correct += 1
        else:
            api_abnormal.append((i, res_targets[i], res_predictions[i], api_targets[i], api_predictions[i]))
            if res_targets[i] == -1:
                only_fail_abnormal.append(i)
        if res_predictions[i] == res_targets[i]:
            res_correct += 1
        else:
            result_abnormal.append((i, res_targets[i], res_predictions[i]))
    api_accuracy = round(100 * api_correct / n, 4)
    res_accuracy = round(100 * res_correct / n, 4)
    result_abnormal_fail_continuous = get_continuous_data(only_fail_abnormal)
    print("api_accuracy:{}, res_accuracy:{}".format(api_accuracy, res_accuracy))
    print("len(api_abnormal):", len(api_abnormal))
    print("len(result_abnormal):", len(result_abnormal))
    print("len(only_fail_abnormal):", len(only_fail_abnormal))
    print("len(result_abnormal_fail_continuous):", len(result_abnormal_fail_continuous))
    return api_abnormal, result_abnormal, result_abnormal_fail_continuous
