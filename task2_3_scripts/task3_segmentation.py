import os
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Paths
# ---------------------------
START_PATH = 'complete_dataset/'
ANNOTATION_PATH = START_PATH + 'annotations.json'

with open(ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

images = data['images']
train_ids = data['splits']['chessred2k']['train']['image_ids']
val_ids = data['splits']['chessred2k']['val']['image_ids']

train_paths = [START_PATH + images[img_id]["path"] for img_id in train_ids]
val_paths = [START_PATH + images[img_id]["path"] for img_id in val_ids]

# ---------------------------
# Dataset
# ---------------------------
class ChessboardDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [(path, path.replace("images", "masks")) for path in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask

# ---------------------------
# Model
# ---------------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))

        return torch.sigmoid(self.conv_last(d1))

# ---------------------------
# Transforms
# ---------------------------
transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(),
    ToTensorV2()
])

# ---------------------------
# Utils
# ---------------------------
def compute_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return correct / total

# ---------------------------
# Training and Evaluation
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(outputs, masks).item()

    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_acc += compute_accuracy(outputs, masks).item()

    return total_loss / len(loader), total_acc / len(loader)

# ---------------------------
# Main
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ChessboardDataset(train_paths, transform=transform)
    val_dataset = ChessboardDataset(val_paths, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    num_epochs = 10
    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    print("Starting training...\n")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} ------------------------")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        torch.save(model.state_dict(), f"checkpoints/unet_epoch_{epoch+1}.pt")

    # Save final model
    torch.save(model.state_dict(), "unet_final.pt")

    # Plot
    plt.plot(train_loss_hist, label="Train Loss", marker='o')
    plt.plot(val_loss_hist, label="Val Loss", marker='x')
    plt.legend()
    plt.title("Loss")
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.close()

    plt.plot(train_acc_hist, label="Train Acc", marker='o')
    plt.plot(val_acc_hist, label="Val Acc", marker='x')
    plt.legend()
    plt.title("Accuracy")
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    plt.close()

    print("Training complete!")

if __name__ == "__main__":
    main()
