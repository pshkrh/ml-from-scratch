import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

class NaiveBayes:
    def __init__(self):
        self._classes = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self._classes = np.unique(y)
        num_classes = len(self._classes)

        # Calculate mean, variance and prior for each class
        self._mean = np.zeros((num_classes, num_features))
        self._var = np.zeros((num_classes, num_features))
        self._priors = np.zeros(num_classes)

        for idx, cls in enumerate(self._classes):
            x_class = X[y == cls]
            self._mean[idx, :] = x_class.mean(axis=0)
            self._var[idx, :] = x_class.var(axis=0)
            self._priors[idx] = x_class.shape[0] / float(num_samples)


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx, cls in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior += prior
            posteriors.append(posterior)

        # Return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, idx, x):
        mean = self._mean[idx]
        var = self._var[idx]
        return (np.exp(-((x - mean) ** 2) / (2 * var))) / (np.sqrt(2 * np.pi * var))


if __name__ == '__main__':
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, n_clusters_per_class=1, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy(y_test, y_pred)}")