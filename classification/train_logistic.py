


import numpy as np
import matplotlib
matplotlib.use("Agg")
from logistic_model import LogisticRegressionScratch
import matplotlib.pyplot as plt


np.random.seed(42)

n = 200

X_class0 = np.random.randn(n,2) + np.array([-2,-2])
X_class1 = np.random.randn(n,2) + np.array([2,2])

X = np.vstack((X_class0, X_class1))

y = np.array([0]*n + [1]*n)

model = LogisticRegressionScratch(lr=0.1, epochs=1000)

model.fit(X,y)

preds = model.predict(X)

accuracy = np.mean(preds == y)

print("\nAccuracy:", accuracy)
print("Weights:", model.w)
print("Bias:", model.b)

plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr")

x_values = np.linspace(X[:,0].min(), X[:,0].max(), 100)

y_values = -(model.w[0]*x_values + model.b) / model.w[1]

plt.plot(x_values, y_values)

plt.title("Decision Boundary")
plt.savefig("decision_boundary.png")
print("Plot saved as decision_boundary.png")