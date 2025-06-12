import os
import json
import numpy as np

# === Paths ===
ANNOTATION_PATH = 'complete_dataset/annotations.json'
LABELS_DIR = 'complete_dataset/dataset_corners/labels' 
SPLITS_DIR = 'complete_dataset/dataset_corners/splits'   
FULL_DATASET = True

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

os.makedirs(LABELS_DIR, exist_ok=True)

KEYPOINT_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]

for img_id, image_info in images.items():
    if img_id not in ann_by_image:
        continue

    ann = ann_by_image[img_id]
    img_width = image_info['width']
    img_height = image_info['height']

    # Get path for label file
    base_name = image_info['path'].replace(".jpg", ".txt").split("images/")[1]
    label_path = os.path.join(LABELS_DIR, base_name)
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    # --- Collect and Sort Corner Coordinates ---
    points = []
    for keypoint_name in KEYPOINT_ORDER:
        coords = ann.get(keypoint_name)
        if coords:
            points.append(coords)

    # Ensure we have all 4 corners
    if len(points) != 4:
        print(f"Warning: Missing corners for image_id {img_id}. Skipping.")
        continue
    

    # TODO: This can be cool to test
    # --- Calculate Bounding Box from Corners ---
    #min_x, min_y = points.min(axis=0)
    #max_x, max_y = points.max(axis=0)
    #box_w = max_x - min_x
    #box_h = max_y - min_y
    #box_cx = min_x + box_w / 2
    #box_cy = min_y + box_h / 2
    # --- Normalize Everything ---
    #box_cx_norm = box_cx / img_width
    #box_cy_norm = box_cy / img_height
    #box_w_norm = box_w / img_width
    #box_h_norm = box_h / img_height

    keypoints_norm = []
    for point in points:
        x_norm = point[0] / img_width
        y_norm = point[1] / img_height
        keypoints_norm.extend([x_norm, y_norm, 2.0]) # x, y, visibility

    box_cx_norm = 0.5
    box_cy_norm = 0.5
    box_w_norm = 1.0
    box_h_norm = 1.0

    # --- Write Label File ---
    with open(label_path, 'w') as f:
        # Class ID for "chessboard" is 0
        class_id = 0
        
        # Write bounding box
        f.write(f"{class_id} {box_cx_norm} {box_cy_norm} {box_w_norm} {box_h_norm} ")
        
        # Write keypoints
        kpt_str = " ".join([f"{coord}" for coord in keypoints_norm])
        f.write(kpt_str + "\n")

print(f"Successfully generated keypoint labels in '{LABELS_DIR}'")

# --- Write Split Files ---
os.makedirs(SPLITS_DIR, exist_ok=True)
image_id_to_file = {img['id']: img['path'] for img in data['images']}

for split in ['train', 'val', 'test']:

    ids = data['splits'][split]['image_ids'] if FULL_DATASET else data['splits']['chessred2k'][split]['image_ids']

    with open(f"{SPLITS_DIR}/{split}.txt", 'w') as f:
        for img_id in ids:
            fname = image_id_to_file.get(img_id)
            if fname:
                f.write(f"/kaggle/input/chessboard-cv/dataset_cornes/images/{fname.split('images/')[1]}\n")

print(f"Successfully generated split files in '{SPLITS_DIR}'")