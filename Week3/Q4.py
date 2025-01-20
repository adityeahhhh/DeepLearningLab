import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.w * x + self.b


x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).view(-1, 1)  # Reshaped to 2D tensor
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]).view(-1, 1)  # Reshaped to 2D tensor
dataset = RegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = RegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

loss_list = []

for epoch in range(100):
    total_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        predictions = model(inputs)

        loss = criterion(predictions, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    loss_list.append(avg_loss)

    print(f"Epoch {epoch + 1}, w={model.w.item()}, b={model.b.item()}, Loss={avg_loss}")

plt.plot(loss_list)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
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