from typing import List
import pandas as pd

from data_prepocessing import DataPreprocessing


def make_submission(
    data_class: DataPreprocessing, predict, features: List[str], scale: bool = False
):
    test_data = data_class.load_test_data(features, scale=scale)

    # make predictions on the test data
    test_prediction = predict(test_data)

    # save the prediction in the appropriate submission format
    test_data_raw = data_class.test_data_raw
    test_data_raw["Survived"] = test_prediction
    submission = test_data_raw[["PassengerId", "Survived"]]
    submission.to_csv("submission.csv", index=False)
