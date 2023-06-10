import pandas as pd
from preprocess.log import Log
import datetime
import pytz
from collections import Counter
import numpy as np


def convert_windows_time(windows_time: str, tz='UTC') -> str:
    """
    Convert 18 Windows filetime to human-readable timestamp
    :param windows_time: 18 Windows filetime
    :param tz: timezone
    :return: formatted timestamp(str)
    """
    epoch_start = datetime.datetime(1601, 1, 1)  # 1601-01-01 00:00:00, 18 Windows filetime start time
    delta = datetime.timedelta(microseconds=int(windows_time) // 10)  # convert 100ns to 1us
    utc_time = epoch_start + delta  # UTC time, standard time zone
    local_tz = pytz.timezone(tz)  # local time zone
    local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(local_tz)  # convert to local time zone
    local_dt = datetime.datetime.fromisoformat(str(local_time))  # convert to datetime.datetime
    formatted_timestamp = local_dt.strftime('%Y-%m-%d %H:%M:%S.%f')  # formattime timestamp
    return formatted_timestamp


def get_tf(word_dic: dict) -> dict:
    """
    Tf = 0.5 + 0.5 * (count / max_tf)
    """
    max_count = max(word_dic.values())
    for k, v in word_dic.items():
        word_dic[k] = 0.5 + 0.5 * (v / max_count)
    return word_dic


def linear_transform(data, a=0, b=5):
    """
    linear transform the data to [a, b]
    :param data:
    :param a:
    :param b:
    :return:
    """
    min_val = np.min(data)
    max_val = np.max(data)
    eps = 10 ** (-7)
    range1 = max_val - min_val + eps
    range2 = b - a
    new_data = [(x - min_val) * (range2 / range1) + a for x in data]
    return new_data


class ApiLog(Log):  # API Logger Log
    """
    ApiLog class is the subclass of Log class.
    AplLog class is used to preprocess the API Logger log file.
    """

    def __init__(self, path, fold=False):
        super(ApiLog, self).__init__(path)
        self.path = path
        self.df = self.get_api_log()  # read log csv file
        self.time_process()  # convert Time column to timestamp
        self.data_process()
        if fold:
            self.df = self.fold()
        self.log_length = len(self.df)

    def __len__(self):
        return len(self.df)

    def get_api_log(self):
        df = pd.read_csv(self.path, low_memory=False, parse_dates=["Time"])
        df.fillna("Na", inplace=True)
        return df

    def time_process(self):
        """
        Convert Time column to timestamp
        """
        start_time = int(self.df["Time"].iloc[0])
        # 修改time列的数据类型为长整形
        self.df["Time_Diff"] = self.df["Time"].astype("int64")
        self.df["Time_Diff"] = self.df["Time_Diff"] - start_time
        # 把time列调整到第一列
        self.df.insert(0, "Time_Diff", self.df.pop("Time_Diff"))
        self.df.insert(0, "Time", self.df.pop("Time"))
        self.df["Time"] = self.df["Time"].apply(convert_windows_time)
        return

    def data_process(self):

        def pr_merge(x):
            str_merge = str(x).split("(")[0].rsplit(" ")[-1]
            return str_merge

        self.df['Prototype'] = self.df['Prototype'].apply(pr_merge)
        self.df.drop(["Sequence"], axis=1, inplace=True)
        return

    def data_factorize(self):
        """
        Factorize the data
        :return: factorized data
        """

        def result_trans(s: str) -> int:
            if s == "OK":
                return 1
            elif s == "FAIL":
                return -1
            return 0

        self.df["Result"] = self.df["Result"].apply(result_trans)
        # self.df["Result"] = pd.factorize(self.df["Result"])[0]
        # 将ProcessId和ThreadId根据值转化为数字变量
        self.df["Process"] = pd.factorize(self.df["ProcessId"])[0]
        self.df["Thread"] = pd.factorize(self.df["ThreadId"])[0]
        self.df["Error"] = pd.factorize(self.df["Error_Code"])[0]
        self.df["Msg"] = pd.factorize(self.df["Msg_Type"])[0]
        self.df["Module"] = pd.factorize(self.df["ModuleName"])[0]
        self.df.drop(["ProcessId", "ThreadId", "Error_Code", "Msg_Type", "ModuleName", "Time"], axis=1, inplace=True)

    def add_tf(self):
        # 除Time_Diff列以外的列全部组合在一起，形成一个新的列，并计算tf
        self.df["merge"] = self.df.apply(lambda x: " ".join([str(x[i]) for i in self.df.columns if i != "Time_Diff"]),
                                         axis=1)
        merge = self.df["merge"].tolist()
        # 因为测试数据只包含一个文档，这里面只计算tf，idf用1代替
        merge_cnt = Counter(merge)
        merge_tf = get_tf(merge_cnt)
        self.df["tfidf"] = self.df["merge"].apply(lambda x: merge_tf[x])
        tfidf = self.df["tfidf"].tolist()
        tfidf = linear_transform(tfidf)
        self.df["tfidf"] = tfidf

    def fold(self):
        """
        将连续出现的相同的行合并成一行
        :return: 合并后的数据集
        """
        df_log = self.df
        # 除time列，其余列拼接成一个字符串
        time_list = df_log["time"].tolist()
        df_log.drop(["time"], axis=1, inplace=True)

        # 定义一个函数，将每行的值拼接成一个新的列
        def merge_rows(row):
            return ','.join(map(str, row))

        # 对新的DataFrame应用函数，创建一个新的merge列
        df_log['merge'] = df_log.apply(merge_rows, axis=1)
        # 将time列和merge列合并成一个新的dataframe
        new_df = pd.DataFrame({"Time": time_list, "merge": df_log["merge"]})
        # 将merge列进行分组，计算连续出现的次数和第一次出现的时间
        df_grouped = new_df.groupby((new_df['merge'] != new_df['merge'].shift(1)).cumsum()).agg(
            {'Time': ['first', 'size'], 'merge': 'first'})
        df_grouped.columns = ['Time', 'count', 'merge']
        # 改变数据格式 把merge的api提取出来，count和别的变量合并，加入时间差变量
        # 加入时间差变量
        time = df_grouped["Time"].tolist()
        time_diff = [0]
        for i in range(1, len(time)):
            time_diff.append(time[i] - time[i - 1])
        df_grouped["time_diff"] = time_diff
        # 提取result变量
        df_grouped["result"] = df_grouped["merge"].apply(lambda x: x.split(",")[0])
        # 提取error变量
        df_grouped["error"] = df_grouped["merge"].apply(lambda x: x.split(",")[1])
        # 提取api变量
        df_grouped["api"] = df_grouped["merge"].apply(lambda x: x.split(",")[2])
        # 提取process变量
        df_grouped["process"] = df_grouped["merge"].apply(lambda x: x.split(",")[3])
        # 提取thread变量
        df_grouped["thread"] = df_grouped["merge"].apply(lambda x: x.split(",")[4])
        # 删除merge列
        df_grouped.drop(["merge"], axis=1, inplace=True)
        # 将result列放在第二列
        cols = list(df_grouped)
        cols.insert(1, cols.pop(cols.index('result')))
        df_grouped = df_grouped.loc[:, cols]
        return df_grouped
