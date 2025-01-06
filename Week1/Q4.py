import numpy as np
import torch

# Create a NumPy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])

# Convert the NumPy array to a PyTorch tensor
tensor_from_numpy = torch.from_numpy(numpy_array)

print("NumPy Array:\n", numpy_array)
print("Converted Tensor:\n", tensor_from_numpy)
