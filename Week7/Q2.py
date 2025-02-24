import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transformations and dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root='/home/student/Desktop/New Folder/220962356/PythonProject/Week6/cats_and_dogs_filtered', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple CNN model for the task
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 2)  # Binary classification (cat vs dog)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel().cuda()


# Custom L1 regularization function
def l1_regularization(model, lambda_l1=0.01):
    l1_norm = 0.0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))  # L1 norm is the sum of absolute values
    return lambda_l1 * l1_norm


# Train loop with L1 regularization
def train_with_l1(epochs=5, lambda_l1=0.01):
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # No weight decay here
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add L1 regularization term to the loss
            l1_loss = l1_regularization(model, lambda_l1)
            total_loss = loss + l1_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


# Train the model with L1 regularization
train_with_l1()


# Custom L1 regularization function (same as above)
def l1_regularization_manual(model, lambda_l1=0.01):
    l1_norm = 0.0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))  # L1 norm: sum of absolute values
    return lambda_l1 * l1_norm


# Train loop with manual L1 regularization
def train_with_manual_l1(epochs=5, lambda_l1=0.01):
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # No weight decay here
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add manual L1 regularization term to the loss
            l1_loss = l1_regularization_manual(model, lambda_l1)
            total_loss = loss + l1_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


# Train the model with manual L1 regularization

# Checking L1 norm of weights before and after training
def print_l1_norm_of_weights(model):
    l1_norm = 0.0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))  # L1 norm: sum of absolute values
    print(f"L1 norm of weights: {l1_norm.item()}")

# Print L1 norm before training
print_l1_norm_of_weights(model)
train_with_manual_l1()
# After training, print L1 norm again to observe the change
print_l1_norm_of_weights(model)
