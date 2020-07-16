from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty",
    "Dona": "Royalty",
}


class DataPreprocessing:
    def __init__(self):
        # load trainings data
        self.raw_data = pd.read_csv("data/train.csv")

        # data preprocessing

        # self.raw_data = self.raw_data.loc[
        #    (self.raw_data["Age"] > 0.0) & (self.raw_data["Age"] <= 70.0)
        #    ]

        self._mean_age = self.raw_data["Age"].mean()
        self._mean_fare = self.raw_data["Fare"].mean()

        self.leS = preprocessing.LabelEncoder()
        self.leT = preprocessing.LabelEncoder()

        self.scaler = preprocessing.StandardScaler()

        self.raw_data = self.data_preprocessing(
            self.raw_data, self.leS.fit_transform, self.leT.fit_transform
        )

        self.test_data_raw = None

    def data_preprocessing(self, data: pd.DataFrame, leS, leT) -> pd.DataFrame:
        data["Sex"] = leS(data["Sex"])

        # create a new feature from the title name
        data["Title"] = data["Name"].apply(self._process_title)

        data["Title"] = leT(data["Title"])

        # create a new feature from the number of siblings and parents aboard the Titanic
        data["Family_Size"] = data["SibSp"] + data["Parch"] + 1

        # fill missing age values with the mean value of all ages
        data["Age"].fillna(self._mean_age, inplace=True)

        # the test data set has one NaN value in the Fare feature
        data["Fare"].fillna(self._mean_fare, inplace=True)
        return data

    def get_data(
            self, features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = self.raw_data[features]

        label = self.raw_data["Survived"]

        return train_test_split(data, label, test_size=0.2)

    def get_raw_data(
            self, features: List[str], scale: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.raw_data[features]
        if scale:
            data = pd.DataFrame(self.scaler.fit_transform(data))

        label = self.raw_data["Survived"]

        return data, label

    def load_test_data(self, features: List[str], scale: bool = False) -> pd.DataFrame:
        # load test data
        self.test_data_raw = pd.read_csv("data/test.csv")

        test_data = self.data_preprocessing(
            self.test_data_raw, self.leS.transform, self.leT.transform
        )

        test_data = test_data[features]

        if scale:
            test_data = pd.DataFrame(self.scaler.transform(test_data))
        return test_data

    def _process_title(self, name: str) -> str:
        n = name.split(",")
        n = n[1].split(".")
        return Title_Dictionary[n[0].strip()]
