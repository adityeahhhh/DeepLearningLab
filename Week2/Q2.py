import torch
import torch.nn.functional as F

b=torch.tensor(1.0,requires_grad=True)
x=torch.tensor(1.0,requires_grad=True)
w=torch.tensor(1.0,requires_grad=True)

u=w*x
v=u+b
a=F.relu(v)

a.backward()

dadw=w.grad

print(dadw)
