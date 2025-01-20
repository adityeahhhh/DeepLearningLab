import torch

x=torch.tensor(1.0, requires_grad=True)

y=8*x**4+3*x**3+7*x**2+6*x+3

y.backward()

print(x.grad)