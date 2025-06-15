import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# ------------------------------
# Configuration
# ------------------------------
batch_size = 32
num_epochs = 10
num_classes = 4
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Data Transforms (No Augmentations)
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Load Annotation JSON
# ------------------------------
START_PATH = 'complete_dataset/'
ANNOTATION_PATH = START_PATH + 'annotations.json'
ORIENTATION_ANNOTATION_PATH = START_PATH + 'orientation_annotations.json'

with open(ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

with open(ORIENTATION_ANNOTATION_PATH, 'r') as f:
    annotations = json.load(f)

images = data['images']
train_ids = data['splits']['chessred2k']['train']['image_ids']
val_ids = data['splits']['chessred2k']['val']['image_ids']

train_paths = [START_PATH + images[img_id]["path"] for img_id in train_ids]
val_paths = [START_PATH + images[img_id]["path"] for img_id in val_ids]

# ------------------------------
# Dataset
# ------------------------------
class ChessboardDataset(Dataset):
    def __init__(self, image_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)

        # Get annotation
        ann = self.annotations.get(img_name)
        if ann is None:
            raise ValueError(f"No annotation found for {img_name}")

        label = int(ann.get('orientation', -1))
        if label not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid label {label} for {img_name}")

        sorted_corners = np.array(ann.get('sorted_corners', []), dtype=np.float32)
        if sorted_corners.shape != (4, 2):
            raise ValueError(f"Invalid corners for {img_name}: {sorted_corners}")

        # Load and warp image
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        margin = 300 # pixels
        padded_w, padded_h = w + 2 * margin, h + 2 * margin

        dst_points = np.float32([
            [margin, margin],
            [margin, padded_h - margin - 1],
            [padded_w - margin - 1, margin],
            [padded_w - margin - 1, padded_h - margin - 1]
        ])

        warp_matrix = cv2.getPerspectiveTransform(sorted_corners, dst_points)
        warped_image = cv2.warpPerspective(img_np, warp_matrix, (padded_w, padded_h))

        # Convert back to PIL and apply transform
        warped_image = Image.fromarray(warped_image)
        if self.transform:
            warped_image = self.transform(warped_image)

        return warped_image, label

# ------------------------------
# Datasets and DataLoaders
# ------------------------------
train_dataset = ChessboardDataset(train_paths, annotations, transform=transform)
val_dataset = ChessboardDataset(val_paths, annotations, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ------------------------------
# Model Setup
# ------------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------------------
# Training and Evaluation
# ------------------------------
def train_model():
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Starting training...\n")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} ------------------------")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_acc = evaluate_model()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save model checkpoint
        checkpoint_path = f"models/checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = "models/final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    # Save Training Loss Plot
    print("\nGenerating and saving training loss plot...")
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig("training_loss.png")
    print("Saved: training_loss.png")

    # Save Accuracy Plot
    print("Generating and saving accuracy plot...")
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', label='Train Accuracy', color='green')
    plt.plot(range(1, num_epochs + 1), val_accuracies, marker='x', label='Val Accuracy', color='orange')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy.png")
    print("Saved: accuracy.png")

    print("\nTraining complete!")

def evaluate_model():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# ------------------------------
# Run Training
# ------------------------------
if __name__ == "__main__":
    train_model()
