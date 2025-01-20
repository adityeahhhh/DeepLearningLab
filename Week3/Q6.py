import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X = torch.tensor([[3.0, 8.0], [4.0, 5.0], [5.0, 7.0], [6.0, 3.0], [2, 1]], dtype=torch.float32)
Y = torch.tensor([-3.7,3.5,2.5, 11.5, 5.7], dtype=torch.float32).view(-1, 1)


class MultipleLinearRegressionModel(nn.Module):
    def __init__(self):
        super(MultipleLinearRegressionModel, self).__init__()

        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


model = MultipleLinearRegressionModel()


model.linear.weight.data.fill_(1.0)
model.linear.bias.data.fill_(0.0)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
loss_list = []

for epoch in range(epochs):

    Y_pred = model(X)

    loss = criterion(Y_pred, Y)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

plt.plot(loss_list)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

X_test = torch.tensor([[3.0, 2.0]], dtype=torch.float32)
model.eval()
with torch.no_grad():
    Y_pred_test = model(X_test)

print(f'Predicted Y for X1=3 and X2=2: {Y_pred_test.item()}')
