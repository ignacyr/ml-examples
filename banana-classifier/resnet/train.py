import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models

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

# Wczytanie wstępnie wytrenowanego modelu ResNet-18
model = models.resnet18(pretrained=True)

# Zmiana ostatniej warstwy, aby dostosować model do problemu klasyfikacji bananów
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Trenowanie modelu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

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

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / (i + 1)}")

# Ewaluacja modelu
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")

# Zapisanie modelu
exp_number = 0
while os.path.exists(f"trainings/exp{exp_number}"):
    exp_number += 1
os.makedirs(f"trainings/exp{exp_number}")
torch.save(model.state_dict(), f"trainings/exp{exp_number}/banana_classifier.pth")
# Save accuracy
with open(f"trainings/exp{exp_number}/accuracy.txt", "w") as f:
    f.write(str(accuracy))
