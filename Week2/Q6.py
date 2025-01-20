import torch

x=torch.tensor(1.0, requires_grad=True)
y=torch.tensor(1.0, requires_grad=True)
z=torch.tensor(1.0, requires_grad=True)

f=torch.tanh(torch.log(1+(2*z*x)/torch.sin(y)))

f.backward()

print(y.grad)