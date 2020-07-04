from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from data_prepocessing import DataPreprocessing
from submission import make_submission

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

data_class = DataPreprocessing()

x_train, x_test, y_train, y_test = data_class.get_data(features)

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

clf.fit(x_train, y_train)

# test the model and compute the accuracy
y_ = clf.predict(x_test)

acc = 0.0
for i, y in enumerate(y_test):
    if y == y_[i]:
        acc += 1

print(f"Accuracy: {acc / float(len(y_))}")

make_submission(data_class, clf.predict, features)
