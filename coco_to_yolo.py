import os
import json

# === Paths ===
ANNOTATION_PATH = 'complete_dataset/annotations.json'
LABELS_DIR = 'complete_dataset/labels'

# === Load annotation ===
with open(ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

images = {img['id']: img for img in data['images']}
annotations = data['annotations']['pieces']

# === Build image_id to annotations map ===
ann_by_image = {}
for ann in annotations:
    if len(ann.keys()) < 5:
        continue  # skip if bbox is missing

    img_id = ann['image_id']
    ann_by_image.setdefault(img_id, []).append(ann)

category_freq = {}

# === Create YOLO .txt label files (mirroring image folder structure) ===
for img_id, image_info in images.items():
    path = image_info['path']  # e.g. chessred2k/images/6/G006_IMG000.jpg
    width = image_info['width']
    height = image_info['height']
    base_name = path.replace(".jpg", ".txt")
    base_name = base_name.split("images/")[1]
    
    label_path = os.path.join(LABELS_DIR, base_name)

    # Ensure label directory exists
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    anns = ann_by_image.get(img_id, [])

    with open(label_path, 'w') as f:
        for ann in anns:
            x, y, w, h = ann['bbox']
            class_id = ann['category_id']  # must be 0-indexed!
            category_freq[class_id] = category_freq.get(class_id, 0) + 1

            # Convert to YOLO format (normalized)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w /= width
            h /= height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# === Print category stats ===
for category_id, freq in category_freq.items():
    print(f"Category {category_id}: {freq} occurrences")

# === Write split .txt files ===
SPLITS_DIR = 'complete_dataset/splits'
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