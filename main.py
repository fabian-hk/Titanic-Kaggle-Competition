import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

# load trainings data
raw_data = pd.read_csv("data/train.csv")

# data preprocessing
raw_data["Sex"].loc[raw_data["Sex"] == "male"] = 0
raw_data["Sex"].loc[raw_data["Sex"] == "female"] = 1

mean_age = raw_data["Age"].mean()
raw_data["Age"].fillna(mean_age, inplace=True)
print(raw_data.isna().any())

data = raw_data[features]

label = raw_data["Survived"]

print(data)
print(label)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

# train model
clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
clf.fit(x_train, y_train)

# test the model and compute the accuracy
y_ = clf.predict(x_test)

acc = 0.0
for i, y in enumerate(y_test):
    if y == y_[i]:
        acc += 1

print(f"Accuracy: {acc / float(len(y_))}")

# load test data
test_data_raw = pd.read_csv("data/test.csv")

# test data preprocessing
test_data_raw["Sex"].loc[test_data_raw["Sex"] == "male"] = 0
test_data_raw["Sex"].loc[test_data_raw["Sex"] == "female"] = 1

test_data_raw.fillna(mean_age, inplace=True)

test_data = test_data_raw[features]

# make predictions on the test data
test_prediction = clf.predict(test_data)

# save the prediction in the appropriate submission format
test_data_raw["Survived"] = test_prediction
submission = test_data_raw[["PassengerId", "Survived"]]
submission.to_csv("submission.csv", index=False)
