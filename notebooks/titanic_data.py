# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python [conda env:ml] *
#     language: python
#     name: conda-env-ml-py
# ---

# # Kaggle Titanic Competition

# ## Imports and helper functions

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# -

def display_all(df):
    with pd.option_context("display.max_rows", 1500, "display.max_columns", 1000): 
        display(df)


# ## Load data

# load training data
train = pd.read_csv("../data/train.csv")

# load test data
test = pd.read_csv("../data/test.csv")

# ## Exploratory data analysis

train.shape

# We have:
# - 891 rows and
# - 12 columns

train.head()

# What kind of data types do we have in the data frame?

train.dtypes

display_all(train)

# Display the amount of NaN data per column

display_all( train.isnull().sum().sort_index()/len(train) )

# ## Data cleaning and feature engineering

# ### Remove labels from train set and merge train and test set for feature engineering

y = train.Survived
train.drop(['Survived'], axis=1, inplace=True)

data = pd.concat([train, test])
data.reset_index(inplace=True)

display_all( data )

# Convert categorical columns and do label encoding

# +
# label encoding
cols_to_encode = ['Sex', 'Cabin', 'Ticket', 'Embarked']
for col in cols_to_encode:
    train[col] = train[col].astype('category').cat.codes

train
# -

# Now inspect the data types again

train.dtypes

# The _Name_ column is still of type object. For now, we will dorp this column.

train.drop('Name', axis=1)

train.describe()

# Replace NaNs with the median value

display_all( train.isna().sum().sort_index()/len(train) )

median_age = train["Age"].median()
train["Age"].fillna(median_age, inplace=True)

corr = train.corr()
print(corr)

f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# # Modeling

# ## Baseline model
#
# The first model we try is a Support Vector Machine.
# The Public Leaderboard (PL) score was 0.77990.
#
# Let's try to reproduce this score using cross validation.

# +
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

# data preprocessing
train["Sex"].loc[train["Sex"] == "male"] = 0
train["Sex"].loc[train["Sex"] == "female"] = 1

mean_age = train["Age"].mean()
train["Age"].fillna(mean_age, inplace=True)
# -

display_all( train )

X = train[features]
print(X.isna().any())

# +
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# train model
clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
clf.fit(X, y)
# -

# training score
accuracy_score(y, clf.predict(X))

# test the model and compute the accuracy
accuracy_score(y_test, clf.predict(x_test))

scores = cross_val_score(clf, X, y, cv=20, n_jobs=-1)
scores

print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))





clf = RandomForestClassifier(n_jobs=-1,
                             n_estimators=1000,
                             max_features='sqrt',
                             min_samples_leaf=4,
                             oob_score=True).fit(x_train, y_train)

accuracy_score(y_test, clf.predict(x_test))

scores = cross_val_score(clf, X, y, cv=20, n_jobs=-1)
scores

print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

test["Survived"] = clf.predict(x_test)
submission = test[["PassengerId", "Survived"]]
submission.to_csv("submission.csv", index=False)

test

len(clf.predict(x_test))


