import torch
import os
from ultralytics import YOLO
import numpy as np


def calculate_corners_metric(all_preds, all_labels):
    # Mean Euclidean Distance between predicted and true corners
    # all_preds and all_labels are (N, 8) where N is batch_size, 8 is (x1,y1,x2,y2,x3,y3,x4,y4)
    # Reshape to (N, 4, 2)
    all_preds_reshaped = all_preds.view(-1, 4, 2)
    all_labels_reshaped = all_labels.view(-1, 4, 2)

    # Calculate Euclidean distance for each corner and then average
    distances = torch.sqrt(torch.sum((all_preds_reshaped - all_labels_reshaped)**2, dim=-1))
    mean_distance_per_corner = torch.mean(distances)
    print(f"Mean distance per corner: {mean_distance_per_corner.item()}")
    return mean_distance_per_corner.item()


def get_all_images_paths(test=True, val=True, splits_path="dataset/splits"):
    result = []
    if test:
        test_path = os.path.join(splits_path, "test.txt")
        if os.path.exists(test_path):
            with open(test_path, 'r') as f:
                result.extend([line.strip() for line in f.readlines()])
    if val:
        val_path = os.path.join(splits_path, "val.txt")
        if os.path.exists(val_path):
            with open(val_path, 'r') as f:
                result.extend([line.strip() for line in f.readlines()])
    return result


def process_image(images, model):
    """
    Alternative function that returns normalized coordinates (0-1 range)
    matching your ground truth format
    """
    predictions = []
    ground_truths = []
    
    for image_path in images:
        print(f"Processing image: {image_path}")

        ground_truth_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(ground_truth_path):
            print(f"Ground truth file not found for {image_path}, skipping.")
            continue
            
        # Read ground truth (already normalized)
        ground_truth_string = open(ground_truth_path, 'r').read().strip()
        _, _, _, _, _, x1, y1, _, x2, y2, _, x3, y3, _, x4, y4, _ = ground_truth_string.split()
        ground_truths.append((float(x1), float(y1), float(x2), float(y2), 
                            float(x3), float(y3), float(x4), float(y4)))

        # Perform inference
        results = model(image_path, verbose=False)

        if len(results) == 0:
            print(f"No results for {image_path}, skipping.")
            continue
            
        result = results[0]
        
        if result.keypoints is None or len(result.keypoints) == 0:
            print(f"No keypoints found for {image_path}, skipping.")
            continue
            
        # Get image dimensions for normalization
        img_height, img_width = result.orig_shape
        
        # Get keypoints and normalize them
        keypoints = result.keypoints.xy[0]  # Get first detection
        
        if len(keypoints) >= 4:
            # Normalize coordinates to 0-1 range
            x1, y1 = keypoints[0].tolist()
            x2, y2 = keypoints[1].tolist()
            x3, y3 = keypoints[2].tolist()
            x4, y4 = keypoints[3].tolist()
            
            # Normalize by image dimensions
            x1_norm, y1_norm = x1 / img_width, y1 / img_height
            x2_norm, y2_norm = x2 / img_width, y2 / img_height
            x3_norm, y3_norm = x3 / img_width, y3 / img_height
            x4_norm, y4_norm = x4 / img_width, y4 / img_height
            
            predictions.append((x1_norm, y1_norm, x2_norm, y2_norm, 
                              x3_norm, y3_norm, x4_norm, y4_norm))
        else:
            print(f"Insufficient keypoints found for {image_path}, skipping.")
            continue

    return predictions, ground_truths


if __name__ == "__main__":
    model = YOLO("models/pose/best_3.pt")
    images = get_all_images_paths(test=True, val=False, splits_path="dataset/splits")
    
    norm_predictions, ground_truths = process_image(images, model)

    pred_tensor = torch.tensor(norm_predictions, dtype=torch.float32)
    gt_tensor = torch.tensor(ground_truths, dtype=torch.float32)
    
    metric = calculate_corners_metric(pred_tensor, gt_tensor)
    print(f"Corner metric (mean Euclidean distance): {metric}")