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

# Define the optimizer with weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Adjust weight_decay as needed

# Loss function
criterion = nn.CrossEntropyLoss()


# Train loop
def train_model(epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


# Train the model with L2 regularization (weight decay)
train_model()


# Custom L2 regularization function
def l2_regularization(model, lambda_l2=0.001):
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.norm(param, p=2) ** 2
    return lambda_l2 * l2_norm


# Train loop with manual L2 regularization
def train_with_manual_l2(epochs=5, lambda_l2=0.001):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add L2 regularization term to the loss
            l2_loss = l2_regularization(model, lambda_l2)
            total_loss = loss + l2_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


# Train the model with manual L2 regularization


# Checking L2 norm of weights before and after training
def print_l2_norm_of_weights(model):
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.norm(param, p=2) ** 2
    print(f"L2 norm of weights: {torch.sqrt(l2_norm).item()}")

# Print L2 norm before training
print_l2_norm_of_weights(model)
train_with_manual_l2()
# After training, print L2 norm again to observe the change
print_l2_norm_of_weights(model)
