import os
import json
import cv2
import numpy as np

# === Paths ===
ANNOTATION_PATH = 'complete_dataset/annotations.json'
MASKS_DIR = 'complete_dataset/dataset_corners/masks' 
FULL_DATASET = False

# === Load annotation ===
with open(ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

images = {img['id']: img for img in data['images']}
annotations = data['annotations']['corners']

ann_by_image = {}
for ann in annotations:
    if 'corners' not in ann or not isinstance(ann['corners'], dict) or len(ann['corners']) != 4:
        continue
    img_id = ann['image_id']
    ann_by_image[img_id] = ann['corners'] 

os.makedirs(MASKS_DIR, exist_ok=True)

for img_id, image_info in images.items():
    if img_id not in ann_by_image:
        continue

    ann = ann_by_image[img_id]
    img_width = image_info['width']
    img_height = image_info['height']

    points = [ann[keypoint_name] for keypoint_name in ["top_left", "top_right", "bottom_right", "bottom_left"] if keypoint_name in ann]

    # Create a grayscale mask image and save it
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    if len(points) == 4:
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], color=255)
    else:
        print(f"Warning: Missing corners for image_id {img_id}. Skipping mask creation.")
        continue

    # Save the mask image
    mask_file_name = image_info['path'].split("images/")[1]
    mask_path = os.path.join(MASKS_DIR, mask_file_name)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, mask)


print(f"Successfully generated grayscale masks in '{MASKS_DIR}' directory.")
