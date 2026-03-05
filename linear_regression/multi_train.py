import numpy as np

# dataset (3 samples, 2 features)
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4]
])

y = np.array([8, 7, 18])

X = (X - X.mean(axis=0)) / X.std(axis=0)

# initialize parameters
W = np.random.randn(2)
b = 0

learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):

    # prediction
    y_pred = X @ W + b

    # loss
    loss = np.mean((y - y_pred) ** 2)

    # gradients
    dW = (-2/n) * (X.T @ (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    # update
    W = W - learning_rate * dW
    b = b - learning_rate * db

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {loss}")

print("\nFinal weights:", W)
print("Final bias:", b)