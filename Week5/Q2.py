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

conv2d_layer = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)


outimage_conv2d = conv2d_layer(image)
print("outimage_conv2d =", outimage_conv2d.shape)
