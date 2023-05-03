import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


# Przygotowanie danych
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = ImageFolder(root="../bananas/training", transform=data_transforms)
test_data = ImageFolder(root="../bananas/testing", transform=data_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Budowanie modelu
class BananaClassifier(nn.Module):
    def __init__(self):
        super(BananaClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 1 * 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = BananaClassifier()

# Trenowanie modelu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch { epoch + 1}/{num_epochs}, Loss: {running_loss / (i + 1)}")


# Ewaluacja modelu
def get_exp_dir():
    base_dir = "runs"
    exp_num = 0
    while True:
        exp_dir = os.path.join(base_dir, f"exp{exp_num}")
        if not os.path.exists(exp_dir):
            os.makedirs(os.path.join(exp_dir, "green"))
            os.makedirs(os.path.join(exp_dir, "ripe"))
            return exp_dir
        exp_num += 1

exp_dir = get_exp_dir()

# Ewaluacja modelu
correct = 0
total = 0
model.eval()

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for idx, pred in enumerate(predicted):
            image_path = test_data.imgs[i * 32 + idx][0]
            image_filename = os.path.basename(image_path)
            if pred == 0:  # green
                shutil.copy(image_path, os.path.join(exp_dir, "green", image_filename))
            elif pred == 1:  # ripe
                shutil.copy(image_path, os.path.join(exp_dir, "ripe", image_filename))

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")
