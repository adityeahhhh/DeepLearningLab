import torch
import matplotlib.pyplot as plt

x = torch.tensor( [2,4])
y = torch.tensor( [20,40])

lr=torch.tensor(0.001)

b=torch.tensor([1.0],requires_grad=True)
w=torch.tensor([1.0],requires_grad=True)

loss_list=[]

for epochs in range(2):
    loss=0.0
    for j in range(len(x)):
        a=w*x[j]
        y_p=a+b
        loss+=(y_p-y[j])**2
    loss=loss/len(x)
    loss_list.append(loss.item())

    loss.backward()
    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
    w.grad.zero_()
    b.grad.zero_()
    print(f"The parameters are w={w},b={b} and loss={loss}")
plt.plot(loss_list)
plt.show()

