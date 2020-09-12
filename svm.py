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

# do cross validation
score = cross_val_score(clf, x, y, cv=5)
print(f"Score: {score}, Mean: {np.mean(score)}")

clf.fit(x, y)

make_submission(data_class, clf.predict, features)
