import torch

# Create two random tensors of shape (2, 3) using uniform distribution (0 to 1)
tensor_1 = torch.rand(2, 3)
tensor_2 = torch.rand(2, 3)

# Move both tensors to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if GPU is available
tensor_1 = tensor_1.to(device)
tensor_2 = tensor_2.to(device)

# Print the tensors to confirm they are on the GPU
print("Tensor 1 on GPU:\n", tensor_1)
print("\nTensor 2 on GPU:\n", tensor_2)

# Optionally, check the device of the tensors
print("\nDevice of Tensor 1:", tensor_1.device)
print("Device of Tensor 2:", tensor_2.device)
