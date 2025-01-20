import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

a = 2 * x
b = torch.sin(y)
c = a / b
d = c * z
e = torch.log(d+1)
f = torch.tanh(e)

f.backward()
    
print("Gradient with respect to y:", y.grad)
print("Intermediate values:")
print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")

