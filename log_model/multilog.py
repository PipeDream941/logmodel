import math
from collections import Counter, defaultdict

from .apilog import ApiLog
import numpy as np


def linear_transform(data, a=0, b=10):
    """
    linear transform the data to [a, b]
    :param data:
    :param a:
    :param b:
    :return:
    """
    min_val = np.min(data)
    max_val = np.max(data)
    range1 = max_val - min_val
    range2 = b - a
    new_data = [(x - min_val) * (range2 / range1) + a for x in data]
    return new_data


class MultiLog:
    """
    MultiLog class is the class for multiple log files.
    We get the api_set from all the log files.
    """

    def __init__(self, path):
        self.path = path
        self.file_list = list(self.path.glob("*.csv"))
        self.file_num = len(self.file_list)
        self.df_list = self.get_df_list()
        self.tf_dic_list, self.tf_dic = self.get_api_tf_dic()
        self.api_set = set(self.tf_dic.keys())
        self.df_list = self.add_tfidf()

    def get_df_list(self):
        df_list = []
        for file in self.file_list:
            df_list.append(ApiLog(file).df)
        return df_list

    def __len__(self):
        return self.file_num

    def get_api_tf_dic(self):
        """
        tf_dic_list record the word's occurrence in each file
        :return:
        """
        tf_dic = Counter()
        tf_dic_list = []
        for df in self.df_list:
            cnt = Counter(df["Prototype"].tolist())
            tf_dic_list.append(cnt)
            tf_dic += cnt
        return tf_dic_list, tf_dic

    def get_idf_dic(self):
        """
        idf_dic record the word's occurrence in all the file
        :return:
        """
        idf_dic = defaultdict(int)
        for api in self.api_set:
            for tf_dic in self.tf_dic_list:
                if api in tf_dic:
                    idf_dic[api] += 1
        return idf_dic

    def get_tf(self, word_dic: dict) -> dict:
        """
        Tf = 0.5 + 0.5 * (count / max_tf)
        """
        max_count = max(word_dic.values())
        for k, v in word_dic.items():
            word_dic[k] = 0.5 + 0.5 * (v / max_count)
        return word_dic

    def add_tfidf(self):
        """
        add tfidf column to the df_list
        :return:
        """
        idf_dic = self.get_idf_dic()
        for df in self.df_list:
            api_list = df["Prototype"].tolist()
            tf_dic = Counter(api_list)
            idf_list = [math.log(1 + self.file_num / idf_dic[api]) + 1 for api in api_list]
            tf_list = [tf_dic[api] for api in api_list]
            tfidf_list = [tf * idf for tf, idf in zip(tf_list, idf_list)]
            tfidf_list = linear_transform(tfidf_list, 0, 10)
            df["tfidf"] = tfidf_list
        return self.df_list
