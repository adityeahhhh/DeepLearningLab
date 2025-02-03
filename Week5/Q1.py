import torch
import torch.nn.functional as F
image = torch.rand(6,6)
print("image=", image)
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
print("image=", image)
kernel = torch.ones(3,3)

print("kernel=", kernel)
kernel = kernel.unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0)

outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("outimage=", outimage)