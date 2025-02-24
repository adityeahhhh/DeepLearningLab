import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Define transformations and dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root='/home/student/Desktop/New Folder/220962356/PythonProject/Week6/cats_and_dogs_filtered/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root='/home/student/Desktop/New Folder/220962356/PythonProject/Week6/cats_and_dogs_filtered/validation', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class CNNModelWithoutDropout(nn.Module):
    def __init__(self):
        super(CNNModelWithoutDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification (cat vs dog)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModelWithDropout(nn.Module):
    def __init__(self):
        super(CNNModelWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification (cat vs dog)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.35)  # Dropout layer with a 50% chance of dropping a unit

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout to the fully connected layer
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, epochs=5, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        # Train phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_losses.append(running_train_loss / len(train_loader))
        train_acc.append(100 * correct_train / total_train)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_losses.append(running_val_loss / len(val_loader))
        val_acc.append(100 * correct_val / total_val)

        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Train Acc: {train_acc[-1]:.2f}%, Val Acc: {val_acc[-1]:.2f}%")

    return train_losses, val_losses, train_acc, val_acc
# Initialize models
model_with_dropout = CNNModelWithDropout().cuda()
model_without_dropout = CNNModelWithoutDropout().cuda()

# Train model with dropout
print("Training model with dropout:")
train_losses_dropout, val_losses_dropout, train_acc_dropout, val_acc_dropout = train_model(model_with_dropout, train_loader, val_loader, epochs=5)

# Train model without dropout
print("\nTraining model without dropout:")
train_losses_no_dropout, val_losses_no_dropout, train_acc_no_dropout, val_acc_no_dropout = train_model(model_without_dropout, train_loader, val_loader, epochs=5)
# Plotting loss curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses_dropout, label="Train Loss (With Dropout)")
plt.plot(val_losses_dropout, label="Val Loss (With Dropout)")
plt.plot(train_losses_no_dropout, label="Train Loss (No Dropout)")
plt.plot(val_losses_no_dropout, label="Val Loss (No Dropout)")
plt.title("Loss Comparison (With and Without Dropout)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plotting accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_acc_dropout, label="Train Acc (With Dropout)")
plt.plot(val_acc_dropout, label="Val Acc (With Dropout)")
plt.plot(train_acc_no_dropout, label="Train Acc (No Dropout)")
plt.plot(val_acc_no_dropout, label="Val Acc (No Dropout)")
plt.title("Accuracy Comparison (With and Without Dropout)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

'''Training model with dropout:
Epoch 1, Train Loss: 0.6989, Val Loss: 0.6902, Train Acc: 52.85%, Val Acc: 55.00%
Epoch 2, Train Loss: 0.6876, Val Loss: 0.6843, Train Acc: 54.30%, Val Acc: 57.30%
Epoch 3, Train Loss: 0.6621, Val Loss: 0.6687, Train Acc: 61.65%, Val Acc: 63.80%
Epoch 4, Train Loss: 0.6173, Val Loss: 0.6116, Train Acc: 67.45%, Val Acc: 65.30%
Epoch 5, Train Loss: 0.5803, Val Loss: 0.5715, Train Acc: 69.40%, Val Acc: 70.90%

Training model without dropout:
Epoch 1, Train Loss: 0.7115, Val Loss: 0.6916, Train Acc: 50.35%, Val Acc: 50.00%
Epoch 2, Train Loss: 0.6855, Val Loss: 0.6836, Train Acc: 54.20%, Val Acc: 56.40%
Epoch 3, Train Loss: 0.6615, Val Loss: 0.6392, Train Acc: 61.30%, Val Acc: 61.40%
Epoch 4, Train Loss: 0.5869, Val Loss: 0.6235, Train Acc: 69.35%, Val Acc: 67.70%
Epoch 5, Train Loss: 0.5528, Val Loss: 0.6320, Train Acc: 72.40%, Val Acc: 67.70%'''