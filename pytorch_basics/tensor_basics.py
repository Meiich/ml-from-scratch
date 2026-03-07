import torch

# scalar
scalar = torch.tensor(5)
print("Scalar:", scalar)
print("Scalar shape:", scalar.shape)

# vector
vector = torch.tensor([1, 2, 3])
print("\nVector:", vector)
print("Vector shape:", vector.shape)

# matrix
matrix = torch.tensor([[1, 2], [3, 4]])
print("\nMatrix:\n", matrix)
print("Matrix shape:", matrix.shape)

# zeros and ones
zeros = torch.zeros((2, 3))
ones = torch.ones((2, 3))
print("\nZeros:\n", zeros)
print("\nOnes:\n", ones)

# random tensor
random_tensor = torch.rand((2, 2))
print("\nRandom tensor:\n", random_tensor)

# basic operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("\nAddition:", a + b)
print("Multiplication:", a * b)
print("Dot product:", torch.dot(a, b))

# matrix multiplication
x = torch.tensor([[1.0, 2.0]])
w = torch.tensor([[3.0], [4.0]])

print("\nMatrix multiplication:", torch.matmul(x, w))
