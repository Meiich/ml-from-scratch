import numpy as np

class LogisticRegressionScratch:

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):

            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                loss = -np.mean(
                    y * np.log(y_pred + 1e-9) +
                    (1-y) * np.log(1 - y_pred + 1e-9)
                )
                print(f"epoch {epoch}, loss {loss}")

    def predict(self, X):

        z = np.dot(X, self.w) + self.b
        probs = self.sigmoid(z)

        return (probs >= 0.5).astype(int)