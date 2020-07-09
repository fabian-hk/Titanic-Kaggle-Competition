from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessing:
    def __init__(self):
        # load trainings data
        self.raw_data = pd.read_csv("data/train.csv")

        # data preprocessing
        self.raw_data = self.raw_data.loc[
            (self.raw_data["Age"] > 0.0) & (self.raw_data["Age"] <= 70.0)
            ]

        self.raw_data["Sex"].loc[self.raw_data["Sex"] == "male"] = 0
        self.raw_data["Sex"].loc[self.raw_data["Sex"] == "female"] = 1
        self.raw_data["Sex"] = self.raw_data["Sex"].astype(dtype=np.float, copy=False)

        self._mean_age = self.raw_data["Age"].mean()
        self.raw_data["Age"].fillna(self._mean_age, inplace=True)
        print(self.raw_data.isna().any())

    def get_data(
            self, features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = self.raw_data[features]

        label = self.raw_data["Survived"]

        return train_test_split(data, label, test_size=0.2)

    def get_raw_data(self, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.raw_data[features]
        label = self.raw_data["Survived"]

        return data, label

    def load_test_data(self) -> pd.DataFrame:
        # load test data
        test_data_raw = pd.read_csv("data/test.csv")

        # test data preprocessing
        test_data_raw["Sex"].loc[test_data_raw["Sex"] == "male"] = 0
        test_data_raw["Sex"].loc[test_data_raw["Sex"] == "female"] = 1

        return test_data_raw.fillna(self._mean_age)
