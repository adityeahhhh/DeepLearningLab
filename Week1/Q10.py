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

# Find the index of the maximum and minimum values (flattened index)
max_index_flat = torch.argmax(result)
min_index_flat = torch.argmin(result)

# Convert the flattened index to multi-dimensional index (row, column)
max_index = torch.unravel_index(max_index_flat, result.shape)
min_index = torch.unravel_index(min_index_flat, result.shape)

# Print the maximum and minimum indices
print("\nIndex of Maximum Value:", max_index)
print("Index of Minimum Value:", min_index)

