import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        self.p = p  # Dropout probability, p=0.5 means 50% probability to keep a unit

    def forward(self, x, training=True):
        if training:
            # Generate Bernoulli random mask
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))  # (1 - p) probability to keep a unit
            mask = mask.to(x.device)  # Ensure it's on the correct device (CUDA/CPU)
            # Apply dropout by multiplying the mask
            x = x * mask
            # Scale the output to maintain expected sum of activations
            x = x / (1 - self.p)
        return x

class CNNModelWithDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CNNModelWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 2)  # Binary classification (cat vs dog)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p)  # Use PyTorch's built-in dropout layer

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply built-in dropout
        x = self.fc2(x)
        return x


class CNNModelWithCustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CNNModelWithCustomDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 2)  # Binary classification (cat vs dog)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = CustomDropout(p)  # Use custom dropout layer

    def forward(self, x, training=True):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x, training)  # Apply custom dropout during training
        x = self.fc2(x)
        return x

# Initialize models
model_with_custom_dropout = CNNModelWithCustomDropout(p=0.5).cuda()
model_with_library_dropout = CNNModelWithDropout().cuda()

# Train model with custom dropout
print("Training model with custom dropout:")
train_losses_custom, val_losses_custom, train_acc_custom, val_acc_custom = train_model(model_with_custom_dropout, train_loader, val_loader, epochs=5)

# Train model with built-in dropout
print("\nTraining model with library dropout:")
train_losses_library, val_losses_library, train_acc_library, val_acc_library = train_model(model_with_library_dropout, train_loader, val_loader, epochs=5)

# Plotting loss curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses_custom, label="Train Loss (Custom Dropout)")
plt.plot(val_losses_custom, label="Val Loss (Custom Dropout)")
plt.plot(train_losses_library, label="Train Loss (Library Dropout)")
plt.plot(val_losses_library, label="Val Loss (Library Dropout)")
plt.title("Loss Comparison (Custom vs Library Dropout)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plotting accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_acc_custom, label="Train Acc (Custom Dropout)")
plt.plot(val_acc_custom, label="Val Acc (Custom Dropout)")
plt.plot(train_acc_library, label="Train Acc (Library Dropout)")
plt.plot(val_acc_library, label="Val Acc (Library Dropout)")
plt.title("Accuracy Comparison (Custom vs Library Dropout)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()
