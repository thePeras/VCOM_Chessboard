import os
import json
import cv2
import numpy as np

# === Paths ===
ANNOTATION_PATH = 'complete_dataset/annotations.json'

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

orientation_ann_by_image = {}
for img_id, image_info in images.items():
    if img_id not in ann_by_image:
        continue

    ann = ann_by_image[img_id]
    img_width = image_info['width']
    img_height = image_info['height']

    points = [(ann[keypoint_name] + [keypoint_name]) for keypoint_name in ["top_left", "top_right", "bottom_right", "bottom_left"] if keypoint_name in ann]

    # Sort by y-coordinate
    points.sort(key=lambda p: p[1])
    top_points = points[:2]
    bottom_points = points[2:]

    # Sort top points by x-coordinate
    top_points.sort(key=lambda p: p[0])
    top_left, top_right = top_points
    tp_orientation = top_left[2]

    bottom_points.sort(key=lambda p: p[0])
    bottom_left, bottom_right = bottom_points

    if(tp_orientation == "top_left"):
        orientation = 0 # 0 degrees
    elif(tp_orientation == "top_right"):
        orientation = 1 # 90 degrees
    elif(tp_orientation == "bottom_right"):
        orientation = 2 # 180 degrees
    elif(tp_orientation == "bottom_left"):
        orientation = 3 # 270 degrees

    orientation_ann_by_image[image_info['file_name']] = {
        "image_id": img_id,
        "orientation": orientation,
        "corners": ann,
        "sorted_corners": [top_left[:2], bottom_left[:2], top_right[:2], bottom_right[:2]]
    }

# Save orientation annotations
output_path = 'complete_dataset/orientation_annotations.json'
with open(output_path, 'w') as f:
    json.dump(orientation_ann_by_image, f, indent=4)
print(f"Orientation annotations saved to {output_path}")
