from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessing:
    def __init__(self):
        # load trainings data
        self.raw_data = pd.read_csv("data/train.csv")

        # data preprocessing
        self.raw_data["Sex"].loc[self.raw_data["Sex"] == "male"] = 0
        self.raw_data["Sex"].loc[self.raw_data["Sex"] == "female"] = 1

        self._mean_age = self.raw_data["Age"].mean()
        self.raw_data["Age"].fillna(self._mean_age, inplace=True)
        print(self.raw_data.isna().any())

    def get_data(self, features: List[str]):
        data = self.raw_data[features]

        label = self.raw_data["Survived"]

        return train_test_split(data, label, test_size=0.2)

    def load_test_data(self) -> pd.DataFrame:
        # load test data
        test_data_raw = pd.read_csv("data/test.csv")

        # test data preprocessing
        test_data_raw["Sex"].loc[test_data_raw["Sex"] == "male"] = 0
        test_data_raw["Sex"].loc[test_data_raw["Sex"] == "female"] = 1

        return test_data_raw.fillna(self._mean_age)
