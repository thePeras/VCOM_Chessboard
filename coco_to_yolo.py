import os
import json

# === Paths ===
ANNOTATION_PATH = 'complete_dataset/annotations.json'
LABELS_DIR = 'complete_dataset/corners/labels'

# === Constants for corner annotations ===
POINT_SIZE_NORM = 0.01 

CORNER_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"] 
CORNER_TO_CLASS_ID = {name: i for i, name in enumerate(CORNER_NAMES)}
CLASS_ID_TO_CORNER_NAME = {i: name for i, name in enumerate(CORNER_NAMES)}

# === Load annotation ===
with open(ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

images = {img['id']: img for img in data['images']}
annotations = data['annotations']['corners']

ann_by_image = {}
for ann in annotations:
    if 'corners' not in ann or not isinstance(ann['corners'], dict) or len(ann['corners']) != 4:
        print(f"Skipping annotation id {ann.get('id', 'N/A')} for image_id {ann.get('image_id', 'N/A')} due to missing/invalid 'corners' field.")
        continue

    img_id = ann['image_id']
    ann_by_image.setdefault(img_id, []).append(ann)

category_freq = {}

for img_id, image_info in images.items():
    path = image_info['path']  
    img_width = image_info['width']
    img_height = image_info['height']
    base_name = path.replace(".jpg", ".txt")
    base_name = base_name.split("images/")[1]
    
    label_path = os.path.join(LABELS_DIR, base_name)

    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    anns_for_image = ann_by_image.get(img_id, [])

    with open(label_path, 'w') as f: # ann contains 'corners' dict
            for corner_name, coords in ann['corners'].items():
                if corner_name not in CORNER_TO_CLASS_ID:
                    print(f"Warning: Unknown corner name '{corner_name}' in image_id {img_id}. Skipping this corner.")
                    continue
                
                class_id = CORNER_TO_CLASS_ID[corner_name]
                category_freq[class_id] = category_freq.get(class_id, 0) + 1

                cx, cy = coords

                # Convert to YOLO format (normalized)
                # For corners, (cx, cy) is the center.
                # Width and height are small fixed values.
                x_center_norm = cx / img_width
                y_center_norm = cy / img_height
                w_norm = POINT_SIZE_NORM 
                h_norm = POINT_SIZE_NORM

                f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# === Print category stats ===
print("\nCategory Frequencies for Corners:")
for category_id, freq in sorted(category_freq.items()):
    corner_name = CLASS_ID_TO_CORNER_NAME.get(category_id, f"Unknown ID {category_id}")
    print(f"Category {category_id} ({corner_name}): {freq} occurrences")

# === Write split .txt files ===
SPLITS_DIR = 'complete_dataset/corners/splits'
os.makedirs(SPLITS_DIR, exist_ok=True)

SPLITS = data['splits']['chessred2k']
image_id_to_file = {img['id']: img['path'] for img in data['images']}

for split in ['train', 'val', 'test']:
    ids = SPLITS[split]['image_ids']
    with open(f"{SPLITS_DIR}/{split}.txt", 'w') as f:
        for img_id in ids:
            fname = image_id_to_file.get(img_id)
            if fname:
                f.write(f"complete_dataset/{fname}\n")



