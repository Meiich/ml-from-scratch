import numpy as np
from nn_from_scratch import SimpleNeuralNetwork

np.random.seed(42)

n = 200

X_class0 = np.random.randn(n, 2) + np.array([-2, -2])
X_class1 = np.random.randn(n, 2) + np.array([2, 2])

X = np.vstack((X_class0, X_class1))
y = np.array([0] * n + [1] * n)

model = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, lr=0.1, epochs=1000)
model.fit(X, y)

preds = model.predict(X)
accuracy = np.mean(preds.flatten() == y)

print("\nAccuracy:", accuracy)
print("W1 shape:", model.W1.shape)
print("W2 shape:", model.W2.shape)