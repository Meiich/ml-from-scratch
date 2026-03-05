import numpy as np
import matplotlib.pyplot as plt
from model import LinearRegression

# Generate synthetic data
np.random.seed(42)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Flatten y
y = y.flatten()

# Train model
model = LinearRegression(learning_rate=0.1, n_iters=1000)
model.fit(X, y)

predictions = model.predict(X)

print("Learned weight:", model.weights)
print("Learned bias:", model.bias)

# Plot
plt.scatter(X, y)
plt.plot(X, predictions, color='red')
plt.show()
plt.figure()
plt.plot(model.losses)
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.show()
