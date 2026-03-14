import numpy as np

X= np.array([1, 2, 3, 4])
y = np.array([3,5,7,9])

w= 0
b = 0
learning_rate = 0.01

for _ in range(5000):

    y_pred = w*X + b

    error = y_pred - y

    dw = (2/len(X)) * np.sum(error * X)
    db = (2/len(X)) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db

print(w,b)
