from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

from data_prepocessing import DataPreprocessing
from submission import make_submission

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

data_class = DataPreprocessing()

x, y = data_class.get_raw_data(features)

# train model
clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))

score = cross_val_score(clf, x, y, cv=5)
print(f"Score: {score}, Mean: {np.mean(score)}")

x_train, x_test, y_train, y_test = data_class.get_data(features)
clf.fit(x_train, y_train)

# test the model and compute the accuracy
y_ = clf.predict(x_test)

acc = 0.0
for i, yy in enumerate(y_test):
    if yy == y_[i]:
        acc += 1

print(f"Accuracy: {acc / float(len(y_))}")

make_submission(data_class, clf.predict, features)
