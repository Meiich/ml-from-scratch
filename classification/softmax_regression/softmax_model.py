import numpy as np


class SoftmaxRegressionScratch:
    
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def fit(self, X, y):

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        y_onehot = self.one_hot(y, n_classes)

        for epoch in range(self.epochs):

            scores = np.dot(X, self.W) + self.b
            probs = self.softmax(scores)

            dW = (1/n_samples) * np.dot(X.T, (probs - y_onehot))
            db = (1/n_samples) * np.sum(probs - y_onehot, axis=0)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 100 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
                print(f"epoch {epoch}, loss {loss}")

    def predict(self, X):

        scores = np.dot(X, self.W) + self.b
        probs = self.softmax(scores)

        return np.argmax(probs, axis=1)