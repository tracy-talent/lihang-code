import numpy as np

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

def generate():
    w = (2, 1)
    X = np.arange(0, 10)
    y = X * w[0] + w[1]
    return w, np.concatenate((X, X)), np.concatenate((y + 3, y - 3))

if __name__ == '__main__':
    lm = LogisticModel()
    w, X, y = generate()
    lm.fit(np.concatenate((X[:, np.newaxis], y[:, np.newaxis]), axis=1),
           np.array([1, 0]).repeat(10))
    import matplotlib.pyplot as plt
    px = np.linspace(0, 10, 20)
    py = px * w[0] + w[1]
    ry = -(lm.W[0] * X + lm.W[2]) / lm.W[1]
    plt.figure()
    plt.scatter(X[:10], y[:10], label='1')
    plt.scatter(X[10:], y[10:], label='0')
    plt.plot(px, py, label='genarator')
    plt.plot(X, ry, label='LM')
    plt.legend()
    plt.show()
