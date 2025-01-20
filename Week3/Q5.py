import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9,
                  18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4,
                  19.5, 19.7, 21.2]).view(-1, 1)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6,
                  16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2,
                  17.0, 17.2, 18.6]).view(-1, 1)

model = nn.Linear(in_features=1, out_features=1)


model.weight.data.fill_(1.0)
model.bias.data.fill_(1.0)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 1000
loss_list = []

for epoch in range(epochs):

    y_pred = model(x)

    loss = criterion(y_pred, y)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Weight: {model.weight.data.item()}, Bias: {model.bias.data.item()}')

plt.plot(loss_list)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

with torch.no_grad():
    predicted = model(x)

plt.scatter(x.numpy(), y.numpy(), color='blue', label='Original Data')
plt.plot(x.numpy(), predicted.numpy(), color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
