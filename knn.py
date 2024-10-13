import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self.predict_helper(x) for x in X]
        return predictions

    def predict_helper(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 2], X[:, 3], c=y, edgecolors='k', s=20)
    plt.show()

    model = KNN(k=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"Accuracy: {accuracy}")