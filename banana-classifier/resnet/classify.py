import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models
import os
import shutil

def load_model(model_path):
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def classify(test_folder, model):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data = ImageFolder(root=test_folder, transform=data_transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    class_names = test_data.classes

    exp_number = 0
    while os.path.exists(f"runs/exp{exp_number}"):
        exp_number += 1
    os.makedirs(f"runs/exp{exp_number}/green")
    os.makedirs(f"runs/exp{exp_number}/ripe")

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Zapisanie wyników klasyfikacji
            for i in range(len(inputs)):
                image_path = test_data.samples[total - len(inputs) + i][0]
                image_filename = os.path.basename(image_path)
                predicted_class = class_names[predicted[i]]
                actual_class = class_names[labels[i]]

                if predicted_class == "green":
                    shutil.copy(image_path, f"runs/exp{exp_number}/green/{image_filename}")
                else:
                    shutil.copy(image_path, f"runs/exp{exp_number}/ripe/{image_filename}")

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")
    with open(f"runs/exp{exp_number}/accuracy.txt", "w") as f:
        f.write(str(accuracy))


if __name__ == "__main__":
    model_path = "trainings/exp0/banana_classifier.pth"
    test_folder = "../bananas/testing"

    model = load_model(model_path)
    classify(test_folder, model)
