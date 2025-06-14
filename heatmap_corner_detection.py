import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import cv2

# --- Part 1: Dataset Creation with Heatmap Generation ---
def create_heatmaps(corners_normalized, output_size=(160, 160), sigma=2):
    heatmaps = np.zeros((4, output_size[0], output_size[1]), dtype=np.float32)
    x_range = np.arange(output_size[1])
    y_range = np.arange(output_size[0])
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    for i, (x_norm, y_norm) in enumerate(corners_normalized):
        cx = int(x_norm * output_size[1])
        cy = int(y_norm * output_size[0])
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        exponent = dist_sq / (2.0 * sigma ** 2)
        gaussian = np.exp(-exponent)
        heatmaps[i] = np.maximum(heatmaps[i], gaussian)

    return torch.from_numpy(heatmaps)

class ChessboardHeatmapDataset(Dataset):
    def __init__(self, annotations_path, img_root_dir, transform=None, heatmap_size=(160, 160)):
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.corners_order = ["top_left", "top_right", "bottom_right", "bottom_left"]

        with open(annotations_path, 'r') as f:
            data = json.load(f)

        images_info = {img['id']: img for img in data['images']}
        self.samples = []
        annotations = data['annotations']['corners']

        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in images_info:
                continue

            image_data = images_info[img_id]
            img_path = os.path.join(self.img_root_dir, image_data['path'])
            width, height = image_data['width'], image_data['height']
            
            if not os.path.exists(img_path):
                continue

            corners_dict = ann['corners']
            if len(corners_dict) != 4:
                continue
                
            normalized_coords = []
            for corner_name in self.corners_order:
                x, y = corners_dict[corner_name]
                normalized_coords.append((x / width, y / height))
            
            self.samples.append((img_path, normalized_coords))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, normalized_coords = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        heatmaps = create_heatmaps(normalized_coords, self.heatmap_size)
        
        if self.transform:
            image = self.transform(image)
            
        return image, heatmaps

# --- Part 2: U-Net Model Architecture ---
class UNetWithResnetEncoder(nn.Module):
    """A U-Net model using a pretrained ResNet34 as the encoder."""
    def __init__(self, n_class=4, pretrained=True):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Encoder layers
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        
        self.final_conv = nn.Conv2d(64, n_class, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        # Decoder
        up4 = self.upconv4(enc5)
        dec4 = self.decoder4(torch.cat([up4, enc4], 1))
        
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat([up3, enc3], 1))
        
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], 1))
        
        output = self.final_conv(dec2)
        
        return output

# --- Part 2.5: Ground Truth Visualization ---
def visualize_ground_truth_heatmap(dataset, sample_idx=0):
    img_path, _ = dataset.samples[sample_idx]
    original_image = Image.open(img_path).convert('RGB')
    image_tensor, heatmap_tensor = dataset[sample_idx]
    tensor_height, tensor_width = image_tensor.shape[1], image_tensor.shape[2]
    display_image = original_image.resize((tensor_width, tensor_height))
    combined_heatmap = torch.max(heatmap_tensor, dim=0)[0].cpu().numpy()
    resized_heatmap = cv2.resize(combined_heatmap, (tensor_width, tensor_height), interpolation=cv2.INTER_LINEAR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(display_image)
    ax1.set_title(f'Resized Original Image (Sample #{sample_idx})')
    ax1.axis('off')
    ax2.imshow(display_image)
    ax2.imshow(resized_heatmap, cmap='hot', alpha=0.6)
    ax2.set_title('Image with Correct Ground Truth Overlay')
    ax2.axis('off')
    plt.show()

# --- Part 3: Training and Evaluation Script ---

# --- Hyperparameters and Configuration ---
ANNOTATIONS_FILE = '/kaggle/input/chessboard-cv/annotations.json'
IMAGE_ROOT = '/kaggle/input/chessboard-cv/chessred2k/'
BATCH_SIZE = 16
IMG_SIZE = 640
HEATMAP_SIZE = (IMG_SIZE // 4, IMG_SIZE // 4) # 160x160
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = 'best_chessboard_model.pth'

print(f"Using device: {DEVICE}")
print(f"Input image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Output heatmap size: {HEATMAP_SIZE[0]}x{HEATMAP_SIZE[1]}")

# --- Data Transformations ---
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Dataset and DataLoaders ---
full_dataset = ChessboardHeatmapDataset(
    annotations_path=ANNOTATIONS_FILE,
    img_root_dir=IMAGE_ROOT,
    transform=data_transforms,
    heatmap_size=HEATMAP_SIZE
)

try:
    if len(full_dataset) > 0:
        visualize_ground_truth_heatmap(full_dataset, sample_idx=0)
except Exception as e:
    print(f"Could not visualize ground truth. Error: {e}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

if len(full_dataset) > 0:
    print(f"Found {len(full_dataset)} total samples.")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

# --- Model, Loss, and Optimizer ---
model = UNetWithResnetEncoder(n_class=4).to(DEVICE)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
if len(train_dataset) > 0 and len(val_dataset) > 0:
    print("Starting training...")
    best_val_loss = float('inf') 

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, heatmaps in train_loader:
            images, heatmaps = images.to(DEVICE), heatmaps.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images, heatmaps = images.to(DEVICE), heatmaps.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best model saved to {BEST_MODEL_PATH} (Val Loss: {avg_val_loss:.6f})")

    print("Finished Training!")
else:
    print("Training skipped: Dataset is empty or no validation set.")

# --- Part 4: Inference and Visualization ---
def get_coords_from_heatmaps(heatmaps):
    """Extracts normalized (x, y) coordinates from a batch of heatmaps by finding the max intensity pixel."""
    batch_size, _, h, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(batch_size, 4, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    y_coords = (max_indices // w).float()
    x_coords = (max_indices % w).float()
    # Normalize coordinates to be in [0, 1] range
    x_coords_norm = x_coords / (w - 1)
    y_coords_norm = y_coords / (h - 1)
    coords_normalized = torch.stack([x_coords_norm, y_coords_norm], dim=-1)
    return coords_normalized.cpu().numpy()

def visualize_prediction(image_tensor, predicted_coords_normalized, corners_order):
    """Displays an image and plots the predicted corner locations on top."""
    # De-normalize the image tensor for visualization
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    h, w, _ = img.shape
    colors = ['red', 'green', 'blue', 'yellow']

    predicted_coords_px = predicted_coords_normalized * np.array([w, h])
    
    print("Predicted Corner Coordinates (in pixels):")
    for i, (name, (x, y)) in enumerate(zip(corners_order, predicted_coords_px)):
        print(f"  - {name}: ({x:.2f}, {y:.2f})")
        plt.scatter(x, y, c=colors[i], s=100, label=name, edgecolors='black')

    plt.legend()
    plt.title("Predicted Chessboard Corners on Validation Image")
    plt.axis('off')
    plt.show()

# --- Run Inference using the BEST saved model ---
print(f"\n--- Running Inference on a Sample using '{BEST_MODEL_PATH}' ---")
if val_size > 0 and os.path.exists(BEST_MODEL_PATH):
    inference_model = UNetWithResnetEncoder(n_class=4, pretrained=False).to(DEVICE) # No need for pretrained weights here
    inference_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    inference_model.eval()

    with torch.no_grad():
        sample_image, ground_truth_heatmaps = val_dataset[0]
        predicted_heatmaps = inference_model(sample_image.unsqueeze(0).to(DEVICE))
        predicted_coords = get_coords_from_heatmaps(predicted_heatmaps)
        visualize_prediction(sample_image, predicted_coords[0], full_dataset.corners_order)
else:
    print("Validation set is empty or model file not found, skipping inference example.")
