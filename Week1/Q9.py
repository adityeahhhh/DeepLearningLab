import torch

# Create two random tensors of shape (2, 3) using uniform distribution (0 to 1)
tensor_1 = torch.rand(2, 3)
tensor_2 = torch.rand(2, 3)

# Move both tensors to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if GPU is available
tensor_1 = tensor_1.to(device)
tensor_2 = tensor_2.to(device)

# Perform matrix multiplication by transposing tensor_2 to shape (3, 2)
result = torch.matmul(tensor_1, tensor_2.T)  # tensor_2.T is the transpose of tensor_2

# Print the result of matrix multiplication
print("Result of Matrix Multiplication (2x2):\n", result)

# Find the maximum and minimum values in the result
max_value = torch.max(result)
min_value = torch.min(result)

# Print the maximum and minimum values
print("\nMaximum Value of the Result:", max_value)
print("Minimum Value of the Result:", min_value)

