import pandas as pd
from .log import Log


class ApiLog(Log):  # API Logger Log
    """
    ApiLog class is the subclass of Log class.
    AplLog class is used to preprocess the API Logger log file.
    """

    def __init__(self, path):
        super(ApiLog, self).__init__(path)
        self.df = self.get_apilog()
        self.df = self.data_preposs(self.df)

        self.df = self.fold(self.df)
        self.log_length = len(self.df)

    def get_apilog(self):
        df = pd.read_csv(self.path, low_memory=False, parse_dates=["Time"])
        df.fillna("Na", inplace=True)
        return df


    def data_preposs(self, df_log):
        def pr_merge(x):
            str_merge = str(x).split("(")[0].rsplit(" ")[-1]
            return str_merge

        df_log = df_log.rename(
            columns={'Error_Code': 'Error Code', 'Msg_Type': 'Msg Type'})
        df_log = df_log[(df_log['ModuleName'].str.contains('dll')) | (
            df_log['ModuleName'].str.contains('lpFileName'))]
        df_log = df_log[~df_log['Result'].str.contains('\*')]
        df_log = df_log.fillna("Na")
        df_log['Prototype'] = df_log['Prototype'].apply(pr_merge)

        # df_log['Prototype_Result'] = df_log['Prototype'].map(str)+"_"+df_log['Result'].map(str)
        # df_log =df_log.drop(['Msg Type','Result','Prototype','ProcessId','Time'],axis=1)
        start_time = int(df_log["Time"].iloc[0])
        # 修改time列的数据类型为长整形
        df_log["Time"] = df_log["Time"].astype("int64")
        df_log["time"] = df_log["Time"] - start_time

        # df_log.drop(columns=["Time"], inplace=True)
        df_log.insert(0, "time", df_log.pop("time"))  # 把time列调整到第一列

        df_log.drop(["Sequence", "Msg Type", "Time", "ModuleName"], axis=1, inplace=True)

        # result 结果 OK 1, 其余 0
        def get_result(s):
            return 1 if s == "OK" else 0

        df_log["Result"] = df_log["Result"].apply(get_result)

        # 将ProcessId和ThreadId根据值转化为数字变量
        # 将A列转化为整数
        df_log["Process"] = pd.factorize(df_log["ProcessId"])[0]
        df_log["Thread"] = pd.factorize(df_log["ThreadId"])[0]
        df_log.drop(["ProcessId", "ThreadId"], axis=1, inplace=True)

        return df_log

    def fold(self, df_log):
        """
        将连续出现的相同的行合并成一行
        :param df_log: 数据集
        :return: 合并后的数据集
        """
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
