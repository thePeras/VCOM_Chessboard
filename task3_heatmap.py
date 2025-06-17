import math
import cv2
import numpy as np
from ultralytics import YOLO
import chess
import chess.svg
import io
from PIL import Image
from editdistance import eval as edit_distance
import re
import json
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models, transforms

##==================================== Corner Detection with U-Net Model ====================================================##

class UNetWithResnetEncoder(nn.Module):
    """A U-Net model using a pretrained ResNet34 as the encoder for corner detection."""
    def __init__(self, n_class=4, pretrained=False): # Set pretrained=False for inference
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final_conv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x); enc2 = self.encoder2(enc1); enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3); enc5 = self.encoder5(enc4)
        up4 = self.upconv4(enc5); dec4 = self.decoder4(torch.cat([up4, enc4], 1))
        up3 = self.upconv3(dec4); dec3 = self.decoder3(torch.cat([up3, enc3], 1))
        up2 = self.upconv2(dec3); dec2 = self.decoder2(torch.cat([up2, enc2], 1))
        return self.final_conv(dec2)

def get_coords_from_heatmaps(heatmaps):
    """Extracts normalized (x, y) coordinates from a batch of heatmaps."""
    batch_size, _, h, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(batch_size, 4, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    y_coords = (max_indices // w).float()
    x_coords = (max_indices % w).float()
    x_coords_norm = x_coords / (w - 1)
    y_coords_norm = y_coords / (h - 1)
    return torch.stack([x_coords_norm, y_coords_norm], dim=-1).cpu().numpy()

def predict_board_corners(model, image_path, device, img_size=640):
    """
    Predicts the four inner corners of the chessboard using the U-Net model.
    Returns corners in the order: [top_left, top_right, bottom_right, bottom_left].
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = original_image.size
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_heatmaps = model(image_tensor)
    
    coords_normalized = get_coords_from_heatmaps(predicted_heatmaps)[0]

    corners_pixel = [[int(x_norm * orig_w), int(y_norm * orig_h)] for x_norm, y_norm in coords_normalized]
        
    return corners_pixel

##==================================== Square and Piece Detection Helpers ====================================================##

def get_board_squares(corners, img_color, image_path):
    """
    Performs a perspective warp using the 4 inner corners and calculates
    the 81 grid intersections arithmetically.
    """
    WARPED_IMG_SIZE = 800  

    src_points = np.float32(corners)
    dst_points = np.float32([[0, 0], [WARPED_IMG_SIZE, 0], [WARPED_IMG_SIZE, WARPED_IMG_SIZE], [0, WARPED_IMG_SIZE]])

    warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_color_img = cv2.warpPerspective(img_color, warp_matrix, (WARPED_IMG_SIZE, WARPED_IMG_SIZE))
    
    intersections = []
    square_size = WARPED_IMG_SIZE / 8.0  
    for r in range(9): 
        for c in range(9):  
            x = int(round(c * square_size))
            y = int(round(r * square_size))
            intersections.append((x, y))
            
    found_all_intersections = True
    
    return intersections, found_all_intersections, warp_matrix, warped_color_img

FEN_MAP = {
    "white-pawn": "P", "white-rock": "R", "white-knight": "N", "white-bishop": "B", "white-queen": "Q", "white-king": "K", 
    "black-pawn": "p", "black-rock": "r", "black-knight": "n", "black-bishop": "b", "black-queen": "q", "black-king": "k",
}

def map_pieces_to_board(yolo_result, warp_matrix, fen_map, warped_size=800):
    """ Maps detected pieces to the board using the calculated grid on the warped image. """
    board_matrix = [["*" for _ in range(8)] for _ in range(8)]
    square_size = warped_size / 8.0

    piece_boxes = yolo_result.boxes.xyxy.cpu().numpy()
    piece_classes = yolo_result.boxes.cls.cpu().numpy().astype(int)
    class_names = yolo_result.names

    for i in range(len(piece_boxes)):
        box = piece_boxes[i]
        # Use the center of the base of the bounding box for better piece placement
        xb, yb = (box[0] + box[2]) / 2, box[1] * 0.1 + box[3] * 0.9  
        
        transformed_point = cv2.perspectiveTransform(np.array([[[xb, yb]]], dtype=np.float32), warp_matrix)[0][0]
        tx, ty = transformed_point
        
        if 0 <= tx < warped_size and 0 <= ty < warped_size:
            c = int(tx / square_size)
            r = int(ty / square_size)

            class_name = class_names[piece_classes[i]]
            fen_char = fen_map.get(class_name)
            if fen_char:
                # Avoid overwriting a square if it's already occupied
                if board_matrix[r][c] == "*":
                    board_matrix[r][c] = fen_char
                else:
                    print(f"Warning: Multiple pieces detected in square ({r},{c}). Keeping first one.")

    return board_matrix

def matrix_to_fen(board_matrix):
    fen_rows = []
    for row in board_matrix:
        empty_count = 0
        fen_row = ""
        for cell in row:
            if cell == "*":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

def get_ground_truth_fen(filepath):
    with open('annotations_fen.json', 'r') as f:
        data = json.load(f)
    return data.get(str(filepath.split("/")[-1]), "")

def process_fen(board_matrix, filename):
    pred_fen = matrix_to_fen(board_matrix)
    exp_fen = get_ground_truth_fen(filename).split(' ')[0] 
    pred_string = "".join("".join(row) for row in board_matrix)
    
    # Create comparable string from expected FEN
    exp_string_full = ""
    for char in exp_fen.replace('/', ''):
        if char.isdigit():
            exp_string_full += '*' * int(char)
        else:
            exp_string_full += char
    
    edit_dist = edit_distance(pred_string, exp_string_full)
    return {"pred_fen": pred_fen, "exp_fen": exp_fen, "edit_dist": edit_dist}

# ==================================== Main Execution ====================================================

def process_single_image(filename, piece_model, corner_model, device):
    print(f"\n--- Processing {filename} ---")
    try:
        # 1. Detect inner corners using the U-Net model
        print("Detecting chessboard corners with U-Net model...")
        corners = predict_board_corners(corner_model, filename, device, img_size=640)
        if not corners or len(corners) != 4:
            print("Error: Could not find 4 chessboard corners with model.")
            return None

        # 2. Predict pieces with YOLO
        print("Predicting pieces...")
        yolo_result = piece_model.predict(filename, verbose=False)[0]

        # 3. Rectify perspective and calculate grid
        print("Correcting perspective and calculating grid...")
        img_color = cv2.imread(filename, cv2.IMREAD_COLOR)
        intersections, all_found, warp_matrix, _ = get_board_squares(corners, img_color, filename)

        if not all_found: 
            print("Error: Could not determine the board grid.")
            return None

        # 4. Map pieces to board (already correctly oriented)
        print("Mapping pieces to squares...")
        board_matrix = map_pieces_to_board(yolo_result, warp_matrix, FEN_MAP, warped_size=800)

        # 5. Compute FEN and get edit distance
        print("Calculating FEN and edit distance...")
        fen_results = process_fen(board_matrix, filename)
        return fen_results

    except Exception as e:
        print(f"An unexpected error occurred while processing {filename}: {e}")
        return {"error": str(e)}

def process_list(image_files, piece_model, corner_model, device, mismatches_path):
    edit_distances = []
    csv_headers = ['filename', 'predicted_fen', 'true_fen', 'edit_distance']
    
    csv_lock = threading.Lock()
    
    with open(mismatches_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            future_to_filename = {
                executor.submit(process_single_image, filename, piece_model, corner_model, device): filename 
                for filename in image_files
            }
            
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    img_results = future.result()
                    if img_results is None: continue
                    
                    if "error" in img_results:
                         with csv_lock: csv_writer.writerow([filename, 'ERROR', img_results['error'], -1])
                         continue

                    curr_edit_dist = img_results['edit_dist']
                    with csv_lock:
                        edit_distances.append(curr_edit_dist)
                        if curr_edit_dist != 0:
                            print(f"Prediction error for {filename.split('/')[-1]}! Edit dist: {curr_edit_dist}. Logging.")
                            row_data = [filename, img_results['pred_fen'], img_results['exp_fen'], curr_edit_dist]
                            csv_writer.writerow(row_data)
                        else:
                            print(f"Success for {filename.split('/')[-1]} (edit distance: 0)")
                except Exception as e:
                    print(f"A critical error occurred for future of {filename}: {e}")
                    with csv_lock: csv_writer.writerow([filename, 'CRITICAL_ERROR', str(e), -1])

    if edit_distances:
        mean_dist = np.mean(edit_distances)
        print(f'\nMean edit distance across {len(edit_distances)} images: {mean_dist:.4f}')
    else:
        print("No images were successfully processed.")
    print(f"Mismatch log saved to: {mismatches_path}")

def process_all_images(piece_model, corner_model, device):
    image_directory = IMAGE_DIRECTORY
    mismatches_path = "mismatches/task3/mismatch_log.csv"
    all_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(image_directory)) for f in fn]
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    process_list(image_files, piece_model, corner_model, device, mismatches_path)

def process_mismatches(piece_model, corner_model, device):
    mismatches_path = "mismatches/task3/mismatch_log.csv"
    if not os.path.exists(mismatches_path):
        print(f"No mismatch file found at {mismatches_path}")
        return
    mismatches = pd.read_csv(mismatches_path)
    if mismatches.empty:
        print("No mismatches to re-process.")
        return
    process_list(list(mismatches['filename']), piece_model, corner_model, device, mismatches_path)


if __name__ == '__main__':
    # --- Configuration ---
    PIECE_MODEL_PATH = "models/myYolo11s/weights/best.pt"
    CORNER_MODEL_PATH = 'best_chessboard_model.pth'  
    IMAGE_DIRECTORY = "dataset/images/"
    MISMATCHES_PATH = "mismatches/task3/mismatch_log.csv"

    # --- Model Loading ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load Piece Detection Model (YOLO)
    piece_model = YOLO(PIECE_MODEL_PATH)

    # Load Corner Detection Model (U-Net)
    corner_model = UNetWithResnetEncoder(n_class=4, pretrained=False).to(DEVICE)
    try:
        corner_model.load_state_dict(torch.load(CORNER_MODEL_PATH, map_location=DEVICE))
        corner_model.eval()
        print("Corner detection model loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Corner model not found at '{CORNER_MODEL_PATH}'.")
        print("Please ensure the trained U-Net model file is in the correct location.")
        exit()
    except Exception as e:
        print(f"FATAL ERROR: Could not load corner model. Error: {e}")
        exit()
        
    # --- Execution ---
    # Choose one of the following functions to run:
    
    # To process all images in the dataset directory:
    process_all_images(piece_model, corner_model, DEVICE)
    
    # To re-run only the images that were previously logged as mismatches:
    # process_mismatches(piece_model, corner_model, DEVICE)

    # To process a single image for debugging:
    #single_image_path = "data/images/G000_IMG087.jpg"
    #result = process_single_image(single_image_path, piece_model, corner_model, DEVICE)
    #if result:
    #    print(json.dumps(result, indent=2))