from torch import nn
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn import utils
import numpy as np
import pandas as pd

from data_prepocessing import DataPreprocessing
from submission import make_submission


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        return self.softmax(output)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs):
        self.train()

        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.int64)

        epochs = 6
        for epoch in range(epochs):
            x_tensor, y_tensor = utils.shuffle(x_tensor, y_tensor, random_state=epochs)

            self.optimizer.zero_grad()
            y_ = self(x_tensor)
            loss = self.criterion(y_, y_tensor)
            print(f"Epoch: {epoch}, Train loss: {loss}")

            loss.backward()
            self.optimizer.step()

    def predict(self, x: pd.DataFrame) -> torch.Tensor:
        self.eval()
        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_ = self(x_tensor)
        return torch.argmax(y_, dim=1)

    def evaluate(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        x, y = utils.shuffle(x, y, random_state=10)

        y_ = self.predict(x)

        acc = 0.0
        for y_y, yy in zip(y_, y):
            if y_y == yy:
                acc += 1

        acc = float(acc) / float(len(y))
        print(f"Accuracy: {acc}")
        return acc


features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

data_class = DataPreprocessing()

x, y = data_class.get_raw_data(features, scale=True)

# do cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=16)
score = []
for train_index, test_index in kf.split(x, y):
    x_train, x_test = x.loc[train_index], x.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    torch.manual_seed(10)

    network = NeuralNetwork()
    network.fit(x_train, y_train)
    acc = network.evaluate(x_test, y_test)

    score.append(acc)

print(f"Score: {score}, Mean: {np.mean(score)}")

# train network on the entire data set before submission
torch.manual_seed(10)

network = NeuralNetwork()
network.fit(x, y)

# make the prediction on the test data set for submission
make_submission(data_class, network.predict, features, scale=True)
