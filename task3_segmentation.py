import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import json

# ---------------------------
# 1. Dataset
# ---------------------------
class ChessboardDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory '{image_dir}' does not exist.")
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory '{mask_dir}' does not exist.")

        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    full_image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_image_path, image_dir)
                    full_mask_path = os.path.join(mask_dir, relative_path)
                    self.image_paths.append((full_image_path, full_mask_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_paths[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype('float32')  # binary mask

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # (1, H, W)

        return image, mask

# ---------------------------
# 2. U-Net Model
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
# 3. Transforms
# ---------------------------
train_transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(),
    ToTensorV2()
])

# ---------------------------
# 4. Training Loop
# ---------------------------
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------------------------
# 5. Main
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ChessboardDataset(
        "/kaggle/input/chessboard-corner-detect/images",
        "/kaggle/input/chessboard-corner-detect/masks",
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    num_epochs = 5
    loss_history = []
    accuracy_history = []

    def compute_accuracy(preds, targets, threshold=0.5):
        preds = (preds > threshold).float()
        correct = (preds == targets).float().sum()
        total = torch.numel(preds)
        return correct / total

    def train(model, dataloader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += compute_accuracy(outputs, masks).item()

        avg_loss = running_loss / len(dataloader)
        avg_accuracy = running_accuracy / len(dataloader)
        return avg_loss, avg_accuracy

    for epoch in range(num_epochs):
        print(f"Have started epcoch {epoch+1}/{num_epochs}")
        loss, accuracy = train(model, train_loader, optimizer, criterion, device)
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save model and history
    torch.save(model.state_dict(), "unet_chessboard.pt")

    with open("loss_history.json", "w") as f:
        json.dump(loss_history, f)

    with open("accuracy_history.json", "w") as f:
        json.dump(accuracy_history, f)

    # Plot loss
    plt.plot(loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

    # Plot accuracy
    plt.plot(accuracy_history, marker='o', color='green')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("accuracy_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
