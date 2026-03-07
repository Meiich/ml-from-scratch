import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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

# split
split_idx = int(0.8 * X.size(0))

X_train = X[:split_idx]
y_train = y[:split_idx]

X_test = X[split_idx:]
y_test = y[split_idx:]

# create dataset objects
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model
model = SimpleClassifier()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"epoch {epoch}, avg train loss {total_loss / len(train_loader)}")

# evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        preds = (outputs >= 0.5).float()

        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

test_accuracy = correct / total
print("\nTest Accuracy:", test_accuracy)