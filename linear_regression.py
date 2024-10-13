import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr=0.001, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iter):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/num_samples) * np.dot(X.T, (y_pred - y))
            db = (1/num_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


def rmse(y_test, predictions):
    return np.sqrt(np.mean((y_test - predictions)**2))


if __name__ == '__main__':
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(lr=0.01, num_iter=5000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    error = rmse(y_test, predictions)
    print(f"RMSE: {error}")

    y_pred_line = model.predict(X)
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, s=10)
    m2 = plt.scatter(X_test, y_test, s=10)
    plt.plot(X, y_pred_line, color='black')
    plt.show()
