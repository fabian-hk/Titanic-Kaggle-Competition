import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load trainings data
raw_data = pd.read_csv("data/train.csv")

# data preprocessing
raw_data["Sex"].loc[raw_data["Sex"] == "male"] = 0
raw_data["Sex"].loc[raw_data["Sex"] == "female"] = 1

mean_age = raw_data["Age"].mean()
raw_data["Age"].fillna(mean_age, inplace=True)

f, ax = plt.subplots(figsize=(10,8))
corr = raw_data.corr()
print(corr)

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
