import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_hat = sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1 / num_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / num_samples) * np.sum(y_hat - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self, X):
        y_hat = sigmoid(np.dot(X, self.weights) + self.bias)
        return [1 if y_pred > 0.5 else 0 for y_pred in y_hat]


if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Accuracy: {accuracy}")