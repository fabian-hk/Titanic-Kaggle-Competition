from torch import nn
import torch

from data_prepocessing import DataPreprocessing


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        return self.sigmoid(output)

    def fit(self, x, y, **kwargs):
        self.train()

        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        # y_tensor = nn.functional.one_hot(y_tensor)

        epochs = 5
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_ = self(x_tensor)
            y_ = y_.squeeze()
            loss = self.criterion(y_, y_tensor)
            print(f"Epoch: {epoch}, Train loss: {loss}")

            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        self.eval()
        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_ = self(x_tensor)
        return torch.round(y_)


def scorer(estimator, x, y) -> float:
    return 0.93


features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

data_class = DataPreprocessing()

x_train, x_test, y_train, y_test = data_class.get_data(features)

network = NeuralNetwork()

network.fit(x_train, y_train)

y_ = network.predict(x_test)

acc = 0.0
for y_y, yy in zip(y_, y_test):
    if y_y == yy:
        acc += 1

print(f"Accuracy: {float(acc) / float(len(y_test))}")
