import torch

# Create two random tensors of shape (2, 3) using uniform distribution (0 to 1)
tensor_1 = torch.rand(2, 3)
tensor_2 = torch.rand(2, 3)

# Move both tensors to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if GPU is available
tensor_1 = tensor_1.to(device)
tensor_2 = tensor_2.to(device)

# Print the original tensors
print("Tensor 1 (2x3) on GPU:\n", tensor_1)
print("\nTensor 2 (2x3) on GPU:\n", tensor_2)

# Perform matrix multiplication by transposing tensor_2 to shape (3, 2)
result = torch.matmul(tensor_1, tensor_2.T)  # tensor_2.T is the transpose of tensor_2

# Print the result
print("\nResult of Matrix Multiplication (2x2):\n", result)
