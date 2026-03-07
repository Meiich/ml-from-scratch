import torch
import torch.nn as nn
import torch.optim as optim
from custom_model import SimpleClassifier

# dataset
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])

# model
model = SimpleClassifier()

# loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training
for epoch in range(1000):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss {loss.item()}")

# prediction
with torch.no_grad():
    preds = model(X)
    predicted = (preds >= 0.5).float()

print("\nPredictions:")
print(predicted)