import pandas as pd


class Log:
    """
    Log class is the base class of all log classes.(api logger and monitor)
    It is used to read the log file and judge the type of log file.
    """

    def __init__(self, path):
        self.path = path
        self.df = self.get_log()
        self.log_type = self.data_judge()

    def get_log(self):
        df = pd.read_csv(self.path, low_memory=False)
        df.fillna("Na", inplace=True)
        return df

    def data_judge(self) -> str:
        """judge the type of log through the column name
        Returns:
            str: "Process Monitor" or "API Logger
        """
        if 'Process Name' in self.df.columns:
            return "Process Monitor"
        elif 'ModuleName' in self.df.columns:
            return "API Logger"
        return "Error"
