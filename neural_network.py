import os
import csv
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, 'facial_expressions', 'data', 'legend.csv')
images_dir = os.path.join(script_dir, 'facial_expressions', 'images')

filenames, emotions = [], []
with open(csv_file_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filenames.append(row['image'])
        emotions.append(row['emotion'].lower())

# encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(emotions)

# load images as grayscale arrays
X = []
for fname in filenames:
  img_path = os.path.join(images_dir, fname)
  img = Image.open(img_path).convert('L')
  img = img.resize((48, 48))
  X.append(np.array(img) / 255.0)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

class EmotionDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X).unsqueeze(1)
    self.y = torch.tensor(y)
  def __len__(self):
    return len(self.y)
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

train_dataset = DataLoader(EmotionDataset(X_train, y_train), batch_size=32, shuffle=True)
test_dataset = DataLoader(EmotionDataset(X_test, y_test), batch_size=32)

class CNN(nn.Module):
  def __init__(self, num_classes=len(label_encoder.classes_)):
    super(CNN, self).__init__()

    # Convolutional Block 1
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.dropout1 = nn.Dropout(0.25)

    # Convolutional Block 2
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.dropout2 = nn.Dropout(0.25)

    # Convolutional Block 3
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.pool3 = nn.MaxPool2d(2, 2)
    self.dropout3 = nn.Dropout(0.25)

    # Fully Connected Layers
    self.fc1 = nn.Linear(128*6*6, 256)
    self.bn_fc1 = nn.BatchNorm1d(256)
    self.dropout_fc1 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(256, num_classes)
  
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.pool1(x)
    x = self.dropout1(x)

    x = F.relu(self.bn2(self.conv2(x)))
    x = self.pool2(x)
    x = self.dropout2(x)

    x = F.relu(self.bn3(self.conv3(x)))
    x = self.pool3(x)
    x = self.dropout3(x)

    x = x.view(x.size(0), -1)

    x = F.relu(self.bn_fc1(self.fc1(x)))
    x = self.dropout_fc1(x)
    x = self.fc2(x)

    return F.log_softmax(x, dim=1)


cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0005)

num_epochs = 25
for epoch in range(num_epochs):
  cnn.train()
  running_loss = 0.0
  for inputs, labels in train_dataset:
    optimizer.zero_grad()
    outputs = cnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataset)}")

cnn.eval()
correct = 0
total = 0
with torch.no_grad():
  for inputs, labels in test_dataset:
    outputs = cnn(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

all_preds = []
all_labels = []

cnn.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataset:
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix")
plt.show()

