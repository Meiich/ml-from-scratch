import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1

print("x:", x)
print("y:", y)

y.backward()

print("dy/dx:", x.grad)