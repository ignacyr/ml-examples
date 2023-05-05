import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Device: {gpu_name}")
else:
    print("Device: CPU")

# Data preparation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = ImageFolder(root="../bananas/train", transform=data_transforms)
val_data = ImageFolder(root="../bananas/val", transform=data_transforms)
test_data = ImageFolder(root="../bananas/test", transform=data_transforms)

# Split the training data into training and validation sets
# train_size = int(0.8 * len(train_data))
# val_size = len(train_data) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

# Load data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# Model creation
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

# Model training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
train_losses = []
val_losses = []
best_val_loss = float("inf")
best_state_dict = None
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
    train_losses.append(running_loss / (i + 1))

    # Validate the model
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")

    # Check if model is better than the best one
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = model.state_dict()


# Save models
exp_number = 0
while os.path.exists(f"trainings/exp{exp_number}"):
    exp_number += 1
os.makedirs(f"trainings/exp{exp_number}/best")
os.makedirs(f"trainings/exp{exp_number}/last")

# Save the best model
torch.save(best_state_dict, f"trainings/exp{exp_number}/best/banana_classifier.pth")
# Save the last model
torch.save(model.state_dict(), f"trainings/exp{exp_number}/last/banana_classifier.pth")


def evaluate(mod: BananaClassifier) -> tuple:
    """ Evaluate the model """
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    mod.eval()
    with torch.no_grad():
        for inp, lab in test_loader:
            out = mod(inp)
            _, predicted = torch.max(out, 1)
            total += lab.size(0)
            correct += (predicted == lab).sum().item()
            all_labels.extend(lab.tolist())
            all_predictions.extend(predicted.tolist())
    # Calculate accuracy
    accuracy = 100 * correct / total
    # Calculate precision, recall and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average="weighted")
    # Return metrics
    return accuracy, precision, recall, f1_score


# Evaluate the last model
acc, prec, rec, f1 = evaluate(mod=model)
print(f"LAST: Accuracy: {acc}%, Precision: {prec}, Recall: {rec}, F1-score: {f1}")
with open(f"trainings/exp{exp_number}/last/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Precision: {prec}\n")
    f.write(f"Recall: {rec}\n")
    f.write(f"F1-score: {f1}\n")

# Evaluate the best model
best_model = BananaClassifier()
best_model.load_state_dict(best_state_dict)
acc, prec, rec, f1 = evaluate(mod=best_model)
print(f"BEST: Accuracy: {acc}%, Precision: {prec}, Recall: {rec}, F1-score: {f1}")
with open(f"trainings/exp{exp_number}/best/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Precision: {prec}\n")
    f.write(f"Recall: {rec}\n")
    f.write(f"F1-score: {f1}\n")


# Create a learning curve plot
plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.savefig(f"trainings/exp{exp_number}/learning_curve.png")
plt.close()
