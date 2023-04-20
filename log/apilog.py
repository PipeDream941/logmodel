import pandas as pd
from .log import Log


class ApiLog(Log):  # API Logger Log
    def __init__(self, path, model_dim=768):
        super(ApiLog, self).__init__(path)
        self.df = self.getApiLog()
        self.model_dim = model_dim
        self.df = self.dataPreposs(self.df)
        self.log_length = len(self.df)

    def getApiLog(self):
        df = pd.read_csv(self.path, low_memory=False, parse_dates=["Time"])
        df.fillna("Na", inplace=True)
        return df

    def dataPreposs(self, df_log):
        def pr_merge(x):
            str_merge = str(x).split("(")[0].rsplit(" ")[-1]
            return str_merge
        df_log = df_log.rename(
            columns={'Error_Code': 'Error Code', 'Msg_Type': 'Msg Type'})
        df_log = df_log[(df_log['ModuleName'].str.contains('dll')) | (
            df_log['ModuleName'].str.contains('lpFileName'))]
        df_log = df_log[~df_log['Result'].str.contains('\*')]
        df_log = df_log.fillna("Na")
        # df_log=df_log[~df_log['Prototype'].str.contains('TlsGetValue|CallWindowProc')]
        df_log['Prototype'] = df_log['Prototype'].apply(pr_merge)

        # df_log['Prototype_Result'] = df_log['Prototype'].map(str)+"_"+df_log['Result'].map(str)
        # df_log =df_log.drop(['Msg Type','Result','Prototype','ProcessId','Time'],axis=1)
        start_time = int(df_log["Time"].iloc[0])
        # 修改time列的数据类型为长整形
        df_log["Time"] = df_log["Time"].astype("int64")
        df_log["time"] = df_log["Time"] - start_time

        # df_log.drop(columns=["Time"], inplace=True)
        df_log.insert(0, "time", df_log.pop("time"))  # 把time列调整到第一列

        return df_log
