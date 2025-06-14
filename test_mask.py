import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the single test image
image_path = "complete_dataset/images/33/G033_IMG000.jpg"
input_img = Image.open(image_path).convert("RGB")

from task3_segmentation import UNet, transforms


# Load model and weights
model = UNet().to(device)
model.load_state_dict(torch.load("unet_chessboard.pt", map_location=device))
model.eval()

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    augmented = transforms(image=img_np)
    img_transformed = augmented['image']  # this is already a tensor if ToTensorV2 is used
    
    # If img_transformed is already a tensor, just:
    return img_transformed.unsqueeze(0).to(device)

def postprocess_output(output, threshold=0.5):
    output = output.squeeze().cpu().detach().numpy()
    mask = (output > threshold).astype(np.uint8) * 255
    return mask



def get_corners_from_mask(mask, max_corners=4):
    """
    Given a binary mask, find up to max_corners corner points.
    Returns list of (x, y) tuples.
    """
    # Convert mask to uint8 binary image for OpenCV
    mask_uint8 = np.uint8(mask > 128) * 255

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Find largest contour assuming it's the chessboard area
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate polygon to simplify contour shape
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If approx has more than max_corners points, pick the 4 farthest points
    # Otherwise, use approx points directly
    points = approx.reshape(-1, 2)

    if len(points) > max_corners:
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        indices = np.argsort(distances)[-max_corners:]
        points = points[indices]

    # Convert to list of tuples
    corners = [(int(x), int(y)) for x, y in points]
    return corners


# Run inference
with torch.no_grad():
    input_tensor = preprocess_image(image_path)
    output = model(input_tensor)
    mask = postprocess_output(output)

# Save predicted mask
mask_img = Image.fromarray(mask)
mask_img.save("predicted_mask.png")

# Detect corners
corners = get_corners_from_mask(mask)

orig_w, orig_h = input_img.size
mask_h, mask_w = mask.shape

scale_x = orig_w / mask_w
scale_y = orig_h / mask_h

scaled_corners = [(int(x * scale_x), int(y * scale_y)) for (x, y) in corners]

print("Detected corners:", scaled_corners)

# Draw corners on the original image
img_cv = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
for (x, y) in scaled_corners:
    cv2.circle(img_cv, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

# Convert back to RGB for displaying with matplotlib
img_with_corners = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Show input image and prediction + corners
plt.subplot(1, 2, 1)
plt.imshow(input_img)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_with_corners)
plt.title("Corners Detected")
plt.axis('off')

plt.show()