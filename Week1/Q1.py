import torch

# 1. Reshaping a Tensor
print("---- Reshaping ----")
tensor_reshape = torch.tensor([[1, 2, 3], [4, 5, 6]])
reshaped_tensor = tensor_reshape.reshape(3, 2)
print("Original Tensor:\n", tensor_reshape)
print("Reshaped Tensor:\n", reshaped_tensor)

# 2. Viewing a Tensor
print("\n---- Viewing ----")
viewed_tensor = tensor_reshape.view(3, 2)
print("Viewed Tensor:\n", viewed_tensor)

# 3. Stacking Tensors
print("\n---- Stacking ----")
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
stacked_tensor = torch.stack((tensor1, tensor2), dim=0)  # Stack along new dimension 0
print("Stacked Tensor (along dim=0):\n", stacked_tensor)

# 4. Squeezing a Tensor
print("\n---- Squeezing ----")
tensor_squeeze = torch.tensor([[[1], [2], [3]]])  # Shape (1, 3, 1)
squeezed_tensor = tensor_squeeze.squeeze()
print("Original Tensor with extra dimensions:\n", tensor_squeeze)
print("Squeezed Tensor:\n", squeezed_tensor)

# 5. Unsqueezing a Tensor
print("\n---- Unsqueezing ----")
tensor_unsqueeze = torch.tensor([1, 2, 3])  # Shape (3,)
unsqueezed_tensor = tensor_unsqueeze.unsqueeze(0)  # Add a dimension at position 0
print("Original Tensor:\n", tensor_unsqueeze)
print("Unsqueezed Tensor (added dimension at position 0):\n", unsqueezed_tensor)
