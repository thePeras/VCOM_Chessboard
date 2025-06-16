# yolov8_num_pieces_eval.py

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score # r2_score added for consistency
from ultralytics import YOLO
import os
import cv2
import json
import pickle # Import pickle for saving results

# --- Configuration ---
YOLO_MODEL_PATH = 'myYolov8x/weights/best.pt'  # Path to your trained YOLOv8x model
COMPLETE_DATASET_ROOT_DIR = 'complete_dataset'
IMAGES_DIR_NAME = 'chessred' # Or 'chessred2k' if that's what your YOLO model was trained on
BATCH_SIZE = 16
NUM_WORKERS = 8
CONF_THRESHOLD = 0.25 # Confidence threshold for YOLO detections. Adjust if needed.
RESULTS_SAVE_DIR = 'results-yolov8x_num_pieces' # Directory to save the pickle file
PARTITION = 'valid'
RESULTS_FILENAME = PARTITION + '.pkl' # Name of the pickle file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device for evaluation.")

def chesspos2number(chesspos):
    col = ord(chesspos[0])-ord('a')
    row = 8 - int(chesspos[1])
    return row, col

class ChessDatasetForYOLO(torch.utils.data.Dataset):
    def __init__(self, root_dir, images_dir_name, partition, transform, use_2k_dataset=False):
        self.anns = json.load(open(os.path.join(root_dir, 'annotations.json')))
        self.root = root_dir
        self.images_dir = os.path.join(root_dir, images_dir_name)
        self.ids = []
        self.file_names = []
        self.original_widths = []
        self.original_heights = []

        for x in self.anns['images']:
            self.file_names.append(x['path'])
            self.ids.append(x['id'])
            self.original_widths.append(x['width'])
            self.original_heights.append(x['height'])

        self.file_names = np.asarray(self.file_names)
        self.original_heights = np.asarray(self.original_heights)
        self.original_widths = np.asarray(self.original_widths)
        self.ids = np.asarray(self.ids)
        self.occupancy_boards = torch.zeros((len(self.file_names), 8, 8))

        for piece in self.anns['annotations']['pieces']:
            idx = np.where(self.ids == piece['image_id'])[0][0]
            row, col = chesspos2number(piece['chessboard_position'])
            self.occupancy_boards[idx][row][col] = 1

        splits = self.anns["splits"]["chessred2k"] if use_2k_dataset else self.anns["splits"]
        if partition == 'train':
            self.split_ids = np.asarray(splits['train']['image_ids']).astype(int)
        elif partition == 'valid':
            self.split_ids = np.asarray(splits['val']['image_ids']).astype(int)
        else:
            self.split_ids = np.asarray(splits['test']['image_ids']).astype(int)

        intersect = np.isin(self.ids, self.split_ids)
        self.split_ids = np.where(intersect)[0]
        self.file_names = self.file_names[self.split_ids]
        self.occupancy_boards = self.occupancy_boards[self.split_ids]
        self.num_pieces = torch.sum(self.occupancy_boards.view(len(self.occupancy_boards), 64), axis=-1)
        self.ids = self.ids[self.split_ids]

        self.transform = transform
        print(f"Number of {partition} images for YOLO evaluation: {len(self.file_names)}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        image_path = os.path.join(self.images_dir, self.file_names[i])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

        num_pieces = self.num_pieces[i]
        image = self.transform(image)

        return image, num_pieces.float()

# Transforms for YOLO input
yolo_inference_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((640, 640)), # Resize to 640x640 as per YOLO config
    transforms.ToDtype(torch.float32, scale=True), # Normalize to [0, 1]
])

# Load Dataset
# We'll use the test dataset for evaluation
test_dataset = ChessDatasetForYOLO(
    root_dir=COMPLETE_DATASET_ROOT_DIR,
    images_dir_name=IMAGES_DIR_NAME,
    partition=PARTITION,
    transform=yolo_inference_transform,
    use_2k_dataset=False,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)

# Load YOLO Model
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to(DEVICE)
    print(f"Successfully loaded YOLO model from {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print(f"Please ensure 'ultralytics' is installed and '{YOLO_MODEL_PATH}' is a valid YOLOv8 model.")
    exit()

# Evaluation Loop
all_predicted_counts = []
all_ground_truth_counts = []

print("\nStarting YOLO inference and evaluation...")
with torch.no_grad():
    for i, (images, gt_num_pieces) in enumerate(test_dataloader):
        # YOLO's predict method can take a list of torch.Tensors
        results = yolo_model.predict(images, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)

        for j, result in enumerate(results):
            num_detected_boxes = len(result.boxes) # Count the number of detected bounding boxes

            all_predicted_counts.append(num_detected_boxes)
            all_ground_truth_counts.append(gt_num_pieces[j].item())

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(test_dataloader)} batches...")

print("\nEvaluation complete.")

# Save Results to Pickle File
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
full_results_path = os.path.join(RESULTS_SAVE_DIR, RESULTS_FILENAME)

results_data = {
    "preds": np.array(all_predicted_counts),
    "true": np.array(all_ground_truth_counts)
}

with open(full_results_path, "wb") as f:
    pickle.dump(results_data, f)

print(f"Prediction results saved to '{full_results_path}'")

# Print MAE and R2 here as well for quick check
yolo_predicted_counts_np = np.array(all_predicted_counts)
true_num_pieces_np = np.array(all_ground_truth_counts)

mae = mean_absolute_error(true_num_pieces_np, yolo_predicted_counts_np)
r2 = r2_score(true_num_pieces_np, yolo_predicted_counts_np)

print(f"\nQuick Check - Mean Absolute Error (MAE): {mae:.4f}")
print(f"Quick Check - R-squared (R2) Score: {r2:.4f}")
