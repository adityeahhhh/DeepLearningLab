import torch

x=torch.tensor(1.0,requires_grad=True)

f=torch.exp(-(x**2+2*x+torch.sin(x)))

f.backward()

print("df/dx:",x.grad)