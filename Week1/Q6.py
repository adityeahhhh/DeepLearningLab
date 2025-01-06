import torch

# Create a random tensor of shape (7, 7)
tensor_a = torch.rand(7, 7)

# Create a random tensor of shape (1, 7)
tensor_b = torch.rand(1, 7)

# Perform matrix multiplication using torch.matmul()
result = torch.matmul(tensor_a, tensor_b.T)  # tensor_b.T is the transpose of tensor_b

print("Tensor A (7x7):\n", tensor_a)
print("\nTensor B (1x7):\n", tensor_b)
print("\nResult of Matrix Multiplication (7x1):\n", result)
