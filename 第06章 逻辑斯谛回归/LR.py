import numpy as np

class LRRegressor:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, X):
        return 1.0 / (1 + np.exp(-X))

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([*d, 1.0])
        return np.array(data_mat)

    # X:2维numpy array，Y:1维numpy array
    def fit(self, X, y):
        X = self.data_matrix(X)
        y = y[:, np.newaxis]
        self.W = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for i in range(self.max_iters):
            for i in range(data_mat.shape[0]):
                grad = data_mat[i][:, np.newaxis] * (y[i] -
                        self.sigmoid(np.dot(data_mat[i], self.W)))
                self.W += self.learning_rate * grad
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
            self.learning_rate, self.max_iter))

    def transform(self, X):
       pred = np.dot(self.data_matrix(X), self.W)
       pred = (pred > 0).astype(np.int32)
       return pred

    def score(self, X, y):
        right = 0
        X_test = self.data_matrix(X)
        for x, y in zip(X_test, y):
            result = np.dot(x, self.W)
            if (result > 0 && y == 1) or (result < 0 && y == 9):
                right += 1
        return right / len(X_test)

if __name__ == '__main__':
    LR = LRRegressor(1e-2)
    X = np.random.normal(size=(10, 5))
    Y = np.random.randint(low=0, high=2, size=(10,))
    LR.fit(X, Y)
    new_X = np.random.normal(size=(3, 5))
    pred = LR.transform(X)
    print(X) 
    print(Y) 
    print(new_X)
    print(pred)

