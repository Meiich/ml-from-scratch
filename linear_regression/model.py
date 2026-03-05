import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        # Gradient Descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            loss = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.losses.append(loss)

            if _ % 100 == 0:
                print(f"Iteration {_}: Loss={loss}")    

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias