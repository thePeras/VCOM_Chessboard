import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os

# --- Model Architecture ---
class UNetWithResnetEncoder(nn.Module):
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

# --- Inference and Visualization Functions ---
def get_coords_from_heatmaps(heatmaps):
    """Extracts (x, y) coordinates from a batch of heatmaps."""
    batch_size, _, h, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(batch_size, 4, -1)
    # Find the index of the maximum value in each heatmap
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)

    y_coords = (max_indices // w).float()
    x_coords = (max_indices % w).float()

    x_coords_norm = x_coords / (w - 1)
    y_coords_norm = y_coords / (h - 1)

    coords_normalized = torch.stack([x_coords_norm, y_coords_norm], dim=-1)
    return coords_normalized.cpu().numpy()

def load_model(model_path, device):
    """Loads the trained model from a .pth file."""
    print(f"Loading model from {model_path}...")
    model = UNetWithResnetEncoder(n_class=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_size=(640, 640)):
    """Loads and preprocesses an image for the model."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image # Return tensor and original PIL image

def unnormalize_image(tensor):
    """Converts a normalized tensor back to a displayable image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    img = std * img + mean
    return np.clip(img, 0, 1)

def visualize_complete_prediction(model, image_tensor, original_image, device):
    """
    Generates and displays a comprehensive visualization of the model's predictions.
    """
    corners_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    colors = ['red', 'lime', 'blue', 'yellow']

    with torch.no_grad():
        predicted_heatmaps = model(image_tensor.to(device))

    # Get final coordinates from heatmaps
    predicted_coords_normalized = get_coords_from_heatmaps(predicted_heatmaps)[0]

    # Prepare images for plotting
    display_image = unnormalize_image(image_tensor)
    heatmap_size = (predicted_heatmaps.shape[2], predicted_heatmaps.shape[3])
    img_h, img_w = display_image.shape[:2]

    # Create a combined heatmap for overlay
    combined_heatmap = torch.max(predicted_heatmaps[0], dim=0)[0].cpu().numpy()
    resized_overlay_heatmap = cv2.resize(combined_heatmap, (img_w, img_h))

    # Create figure
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.2, 4, 4])

    # --- Title ---
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.text(0.5, 0.5, "Chessboard Corner Prediction Analysis", ha='center', va='center', fontsize=24, fontweight='bold')
    title_ax.axis('off')

    # --- Original Image ---
    ax_orig = fig.add_subplot(gs[1, 0:2])
    ax_orig.imshow(display_image)
    ax_orig.set_title("Original Input Image", fontsize=16)
    ax_orig.axis('off')

    # --- Individual Heatmaps ---
    for i in range(4):
        ax = fig.add_subplot(gs[1, 2 + i % 2] if i < 2 else gs[2, 0 + i % 2])
        heatmap = predicted_heatmaps[0, i].cpu().numpy()
        ax.imshow(heatmap, cmap='hot')
        ax.set_title(f"Heatmap: {corners_order[i]}", fontsize=14)
        ax.axis('off')

    # --- Heatmap Overlay ---
    ax_overlay = fig.add_subplot(gs[2, 2])
    ax_overlay.imshow(display_image)
    ax_overlay.imshow(resized_overlay_heatmap, cmap='hot', alpha=0.6)
    ax_overlay.set_title("Combined Heatmap Overlay", fontsize=16)
    ax_overlay.axis('off')

    # --- Final Prediction ---
    ax_final = fig.add_subplot(gs[2, 3])
    ax_final.imshow(display_image)
    ax_final.set_title("Final Predicted Corners", fontsize=16)

    # Plot points and labels
    print("\nPredicted Corner Coordinates (pixel values):")
    for i, (x_norm, y_norm) in enumerate(predicted_coords_normalized):
        px, py = x_norm * img_w, y_norm * img_h
        print(f"  - {corners_order[i]}: ({px:.2f}, {py:.2f})")
        ax_final.scatter(px, py, c=colors[i], s=8, label=corners_order[i], edgecolors='black', linewidth=1)
    
    ax_final.legend()
    ax_final.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    """Main function to run the prediction and visualization."""
    parser = argparse.ArgumentParser(description="Visualize chessboard corner predictions from a U-Net model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model .pth file.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)

    image_tensor, original_pil_image = preprocess_image(args.image_path)

    visualize_complete_prediction(model, image_tensor, original_pil_image, device)


if __name__ == '__main__':
    main()