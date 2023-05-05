import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.nn.functional as F


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


# Load the trained model
model_path = "trainings/exp6/banana_classifier.pth"
model = BananaClassifier()
model.load_state_dict(torch.load(model_path))
model.eval()

# Prepare data
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data = ImageFolder(root="../bananas/testing", transform=data_transforms)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Create output directories
exp_number = 0
while os.path.exists(f"runs/exp{exp_number}"):
    exp_number += 1
os.makedirs(f"runs/exp{exp_number}/green", exist_ok=True)
os.makedirs(f"runs/exp{exp_number}/ripe", exist_ok=True)

# Classify, save results, and compute accuracy
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if predicted == 0:
            class_name = "green"
        else:
            class_name = "ripe"

        input_image = Image.open(test_data.imgs[i][0])
        input_image.save(f"runs/exp{exp_number}/{class_name}/{os.path.basename(test_data.imgs[i][0])}")

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")
with open(f"runs/exp{exp_number}/accuracy.txt", "w") as f:
    f.write(str(accuracy))

print(f"Classification results and accuracy saved in runs/exp{exp_number}/.")
