import torch
import torch.nn as nn
import torch.optim as optim
from custom_model import SimpleClassifier

# reproducibility
torch.manual_seed(42)

# synthetic dataset
n = 200

class0 = torch.randn(n, 2) + torch.tensor([-2.0, -2.0])
class1 = torch.randn(n, 2) + torch.tensor([2.0, 2.0])

X = torch.cat([class0, class1], dim=0)
y = torch.cat([torch.zeros(n, 1), torch.ones(n, 1)], dim=0)

# shuffle dataset
indices = torch.randperm(X.size(0))
X = X[indices]
y = y[indices]

# train/test split
split_idx = int(0.8 * X.size(0))

X_train = X[:split_idx]
y_train = y[:split_idx]

X_test = X[split_idx:]
y_test = y[split_idx:]

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# model
model = SimpleClassifier()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training loop
for epoch in range(1000):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}, train loss {loss.item()}")

# evaluation
with torch.no_grad():
    test_outputs = model(X_test)
    test_preds = (test_outputs >= 0.5).float()
    test_accuracy = (test_preds == y_test).float().mean()

print("\nTest Accuracy:", test_accuracy.item())