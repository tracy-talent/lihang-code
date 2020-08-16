import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class LogisticModel:
    def __init__(self, learning_rate=0.1, max_iters=50):
        self.learning_rate = learning_rate
        self.max_iters = max_iters

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def trans(self, X):
        return np.concatenate((np.array(X), np.ones((len(X), 1))), axis=1)

    def fit(self, X, y):
        X = self.trans(X)
        self.W = np.zeros(X.shape[1])
        for iter in range(self.max_iters):
            for i in range(len(X)):
                e = self.sigmoid(X[i].dot(self.W))
                self.W += self.learning_rate * X[i] * (y[i] - e)

    def transform(self, X):
        X = self.trans(X)
        y = np.zeros(len(X))
        for i in range(len(X)):
            if X[i].dot(self.W) >= 0:
                y[i] = 1
            else:
                y[i] = 0
        return y

    def score(self, X, y):
        X = self.trans(X)
        right = 0
        for i in range(len(X)):
            r = X[i].dot(self.W)
            if r >= 0 and y[i] == 1 or \
               r < 0 and y[i] == 0:
                right += 1
        return right / len(X)


if __name__ == '__main__':
    lm = LogisticModel()
    X, y = datasets.make_classification(n_samples=100, n_features=2,
                                    n_informative=2, n_redundant=0, n_repeated=0, 
                                       n_classes=2, n_clusters_per_class=1)
    lm.fit(X, y)
    ry = -(lm.W[0] * X[:, 0] + lm.W[2]) / lm.W[1]
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(X[:, 0], ry, label='LM')
    plt.legend()
    plt.show()
