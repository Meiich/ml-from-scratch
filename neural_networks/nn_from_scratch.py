import numpy as np


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            output = self.forward(X)

            loss = np.mean((y - output) ** 2)

            d_a2 = output - y
            d_z2 = d_a2 * self.sigmoid_derivative(output)

            d_W2 = np.dot(self.a1.T, d_z2) / n_samples
            d_b2 = np.sum(d_z2, axis=0, keepdims=True) / n_samples

            d_a1 = np.dot(d_z2, self.W2.T)
            d_z1 = d_a1 * self.sigmoid_derivative(self.a1)

            d_W1 = np.dot(X.T, d_z1) / n_samples
            d_b1 = np.sum(d_z1, axis=0, keepdims=True) / n_samples

            self.W2 -= self.lr * d_W2
            self.b2 -= self.lr * d_b2
            self.W1 -= self.lr * d_W1
            self.b1 -= self.lr * d_b1

            if epoch % 100 == 0:
                print(f"epoch {epoch}, loss {loss}")

    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int)