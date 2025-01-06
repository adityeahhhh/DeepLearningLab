import torch

tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]]])

print("Original Tensor:\n", tensor)
print("Original Shape:", tensor.shape)

permuted_tensor = tensor.permute(1, 2, 0)

print("\nPermuted Tensor:\n", permuted_tensor)
print("Permuted Shape:", permuted_tensor.shape)
