import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

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
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

plt.plot(loss_list)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

with torch.no_grad():
    y_pred_test = model(x)

plt.scatter(x.numpy(), y.numpy(), color='blue', label='Original Data')
plt.plot(x.numpy(), y_pred_test.numpy(), color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Logistic Regression Fit')
plt.legend()
plt.show()
