import numpy as np
from softmax_model import SoftmaxRegressionScratch

np.random.seed(42)

n = 200

class0 = np.random.randn(n,2) + np.array([-3,-3])
class1 = np.random.randn(n,2) + np.array([3,0])
class2 = np.random.randn(n,2) + np.array([0,3])

X = np.vstack((class0, class1, class2))

y = np.array([0]*n + [1]*n + [2]*n)

model = SoftmaxRegressionScratch(lr=0.1, epochs=1000)

model.fit(X,y)

preds = model.predict(X)

accuracy = np.mean(preds == y)

print("\nAccuracy:", accuracy)
print("Weights shape:", model.W.shape)
print("Bias shape:", model.b.shape)