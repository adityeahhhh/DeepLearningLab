import torch

# Set the random seed for reproducibility
torch.manual_seed(7)

# Create a random tensor with shape (1, 1, 1, 10)
tensor_1 = torch.rand(1, 1, 1, 10)

# Remove all singleton dimensions to get a tensor of shape (10)
tensor_2 = tensor_1.squeeze()

# Print the first tensor and its shape
print("First Tensor (shape: 1x1x1x10):\n", tensor_1)
print("\nShape of the First Tensor:", tensor_1.shape)

# Print the second tensor (squeezed) and its shape
print("\nSecond Tensor (shape: 10):\n", tensor_2)
print("\nShape of the Second Tensor:", tensor_2.shape)
