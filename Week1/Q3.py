import torch

# Create a 2D tensor (matrix)
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Original Tensor:\n", tensor)

# Access element at row 0, column 1
element = tensor[0, 1]
print("\nElement at (0, 1):", element)
