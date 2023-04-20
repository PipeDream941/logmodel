import pandas as pd
class Log:
    def __init__(self, path):
        self.path = path
        self.df = self.getLog(self.path)
        self.logtype = self.dataJudge(self.df)

    def getLog(self, path):
        df = pd.read_csv(path, low_memory=False)
        df.fillna("Na", inplace=True)
        return df

    def dataJudge(self, df: pd.DataFrame) -> str:
        """judge the type of log through the column name

        Args:
            df (panda.DataFrame): _description_

        Returns:
            str: "Process Monitor" or "API Logger
        """
        if 'Process Name' in df.columns:
            return "Process Monitor"
        elif 'ModuleName' in df.columns:
            return "API Logger"
        return "Error"