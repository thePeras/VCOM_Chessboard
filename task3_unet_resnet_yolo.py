import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from editdistance import eval as edit_distance
import json
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
from torchvision import transforms, models
import torch.nn as nn
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## === LOAD MODELS == ##
sg_model_path = "models/mask/unet_chessboard.pt"
cl_model_path = "models/orientation/orientation.pt"
yolo_model_path = "models/myYolov8x/weights/best.pt"

# Unet for Segmentation of the board
from task3_segmentation import UNet, transform as mask_transforms
sg_model = UNet().to(device)
sg_model.load_state_dict(torch.load(sg_model_path, map_location=device))
sg_model.eval()

# Resnet base CNN for orientation classification
cl_model = models.resnet18(weights=None)
cl_model.fc = nn.Linear(cl_model.fc.in_features, 4)
cl_model.load_state_dict(torch.load(cl_model_path, map_location=device))
cl_model = cl_model.to(device)
cl_model.eval()

ORIENTATION_ANNOTATION_PATH = 'complete_dataset/orientation_annotations.json'
with open(ORIENTATION_ANNOTATION_PATH, 'r') as f:
    orientation_annotations = json.load(f)

# Yolo for object detection and classification
yolo_model = YOLO(yolo_model_path)
yolo_model.model.fuse = lambda *args, **kwargs: yolo_model.model

## === Util Methods == #

def get_board_corners(image_path):
    with torch.inference_mode():
        # Load and resize image for faster model processing
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        img_np = np.array(img)

        # Apply mask transforms
        augmented = mask_transforms(image=img_np)
        img_transformed = augmented['image']
        input_tensor = img_transformed.unsqueeze(0).to(device)

        # Forward pass
        output = sg_model(input_tensor).squeeze().cpu().numpy()
        mask = (output > 0.5).astype(np.uint8)

    # Skip image if mask is empty
    if np.count_nonzero(mask) == 0:
        print(f"Empty mask for {image_path}. Skipping.")
        return None, None

    # Resize mask to original image size for accurate contour detection
    input_img = cv2.imread(image_path)
    if input_img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    orig_h, orig_w = input_img.shape[:2]

    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Find contours in resized mask
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in {image_path}. Skipping.")
        return None, None

    # Largest contour assumed to be the board
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    points = approx.reshape(-1, 2)

    if len(points) < 4:
        print(f"Only {len(points)} corner points found in {image_path}. Skipping.")
        return None, None
    elif len(points) > 4:
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        indices = np.argsort(distances)[-4:]
        points = points[indices]

    # Convert to int tuples
    corners = [(int(x), int(y)) for x, y in points]

    # Sort: top-left, bottom-left, top-right, bottom-right
    sorted_corners = sorted(corners, key=lambda pt: (pt[1], pt[0]))
    top_pts = sorted(sorted_corners[:2], key=lambda pt: pt[0])
    bottom_pts = sorted(sorted_corners[2:], key=lambda pt: pt[0])
    ordered_corners = np.float32([top_pts[0], bottom_pts[0], top_pts[1], bottom_pts[1]])

    return sorted_corners, ordered_corners


def warp_image(image_path, corners, margin=300):
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    padded_w, padded_h = w + 2 * margin, h + 2 * margin

    dst_points = np.float32([
        [margin, margin],
        [margin, padded_h - margin - 1],
        [padded_w - margin - 1, margin],
        [padded_w - margin - 1, padded_h - margin - 1]
    ])

    warp_matrix = cv2.getPerspectiveTransform(corners, dst_points)
    warped_image = cv2.warpPerspective(img_np, warp_matrix, (padded_w, padded_h))

    return Image.fromarray(warped_image), warp_matrix, w, h

def get_board_orientation(image_path, corners):
    warped_img, _, _, _ = warp_image(image_path, corners)

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = transform(warped_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = cl_model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()



##==================================== Horse to detect the chessboard orientation ================================================##

FEN_MAP = {
    "white-pawn": "P",
    "white-rock": "R", 
    "white-knight": "N",
    "white-bishop": "B",
    "white-queen": "Q",
    "white-king": "K", 
    "black-pawn": "p",
    "black-rock": "r",
    "black-knight": "n",
    "black-bishop": "b",
    "black-queen": "q",
    "black-king": "k",
}

def map_pieces_to_board(yolo_result, intersections, warp_matrix, fen_map):
    board_matrix = [["*" for _ in range(8)] for _ in range(8)]
    if not intersections:
        print("No intersections provided, cannot map pieces.")
        return board_matrix

    piece_boxes = yolo_result.boxes.xyxy.cpu().numpy()
    piece_classes = yolo_result.boxes.cls.cpu().numpy().astype(int)
    class_names = yolo_result.names

    grid_side_len = int(math.sqrt(len(intersections)))

    for i in range(len(piece_boxes)):
        box = piece_boxes[i]
        x1, y1, x2, y2 = box
        
        xb = (x1 + x2) / 2
        yb = y1 * 0.1 + y2 * 0.9  

        point_to_transform = np.array([[[xb, yb]]], dtype=np.float32)
        
        transformed_point = cv2.perspectiveTransform(point_to_transform, warp_matrix)
        tx, ty = transformed_point[0][0]
        

        found_square = False
        for r in range(8):
            for c in range(8):
                top_left_idx = r * grid_side_len + c
                top_right_idx = top_left_idx + 1
                bottom_left_idx = (r + 1) * grid_side_len + c
                
                if (intersections[top_left_idx][0] < tx < intersections[top_right_idx][0] and
                    intersections[top_left_idx][1] < ty < intersections[bottom_left_idx][1]):
                    
                    class_name = class_names[piece_classes[i]]
                    fen_char = fen_map.get(class_name)
                    
                    if fen_char:
                        board_matrix[r][c] = fen_char
                    
                    found_square = True
                    break
            if found_square:
                break
    
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
    fen_str = data[str(filepath.split("/")[-1])]
    return fen_str


def process_fen(board_matrix, filename):
    def fen_to_full_str(fen_str):
        result = ""
        for char in fen_str:
            if char.isdigit():
                result += "*" * int(char)
            else:
                result += char
        return result

    pred_fen = matrix_to_fen(board_matrix)
    exp_fen = get_ground_truth_fen(filename)

    pred_string = "/".join("".join(row) for row in board_matrix)
    exp_string = fen_to_full_str(exp_fen)
    
    edit_dist = edit_distance(pred_string, exp_string)
    
    return {"pred_fen":pred_fen, "exp_fen": exp_fen, "edit_dist":edit_dist}

def rotate_matrix(rotation, board_matrix):
    np_matrix = np.array(board_matrix)
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        rotated_matrix = np.rot90(np_matrix, k=-1)
    elif rotation == cv2.ROTATE_180:
        rotated_matrix = np.rot90(np_matrix, k=2)
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        rotated_matrix = np.rot90(np_matrix, k=1)
    else:
        return board_matrix
    return rotated_matrix.tolist()

def get_intersections_and_warp_matrix_from_corners(image_path, corners, board_size=8):
    _, warp_matrix, img_w, img_h = warp_image(image_path, corners, 0)

    # Generate intersections in destination space
    step_x = img_w / board_size
    step_y = img_h / board_size
    intersections = []
    for row in range(board_size + 1):
        for col in range(board_size + 1):
            x = col * step_x
            y = row * step_y
            intersections.append((x, y))

    return intersections, warp_matrix

# ==================================== Main Execution ====================================================

image_directory = "complete_dataset/images/"
mismatches_path = "mismatches/task3/mismatch_log.csv"

def process_single_image(filename):
    print(f"\n--- Processing {filename} ---")

    # 1. Predict pieces
    print("Predicting pieces...")
    results = yolo_model.predict(filename, verbose=False)
    yolo_result = results[0]

    # 2. Detect corners
    print("Detecting chessboard corners...")
    sorted_corners, ordered_corners = get_board_corners(filename)
    if not sorted_corners or len(sorted_corners) != 4:
        print("Error: Could not find chessboard corners. Skipping file.")

    # 3. Extract intersections (board squares)
    print("Calculating warp matrix and intersections...")
    intersections, warp_matrix = get_intersections_and_warp_matrix_from_corners(filename, ordered_corners)

    # 4. Map pieces to board
    print("Mapping pieces to squares...")
    board_matrix = map_pieces_to_board(yolo_result, intersections, warp_matrix, FEN_MAP)

    # 5. Determine orientation
    print("Determining board orientation...")
    orientation = get_board_orientation(filename, ordered_corners)
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    if orientation > 0:
        board_matrix = rotate_matrix(rotations[orientation], board_matrix)

    # 6. Compute FEN and get edit distance
    print("Processing FEN")
    fen_results = process_fen(board_matrix, filename)
    return fen_results


def process_list(image_files):
    edit_distances = []
    csv_headers = ['filename', 'predicted_fen', 'true_fen', 'edit_distance']
    
    # Thread-safe lock for writing to CSV and updating edit_distances
    csv_lock = threading.Lock()
    
    with open(mismatches_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
            # Submit all tasks using the thread-safe wrapper function
            future_to_filename = {executor.submit(process_single_image, filename): filename for filename in image_files}
            
            # Process completed tasks
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    img_results = future.result()
                    if img_results is None:
                        continue
                        
                    curr_edit_dist = img_results['edit_dist']
                    
                    # Thread-safe operations
                    with csv_lock:
                        edit_distances.append(curr_edit_dist)
                        
                        if curr_edit_dist != 0:
                            print(f"Prediction error found for {filename}! Edit distance: {curr_edit_dist}. Logging mismatch.")
                            true_fen = img_results['exp_fen']
                            predicted_fen = img_results['pred_fen']
                            row_data = [filename, predicted_fen, true_fen, curr_edit_dist]
                            csv_writer.writerow(row_data)
                        else:
                            print(f"Successfully processed {filename} (edit distance: 0)")

                except Exception as e:
                    print(f"An unexpected error occurred while processing {filename}: {e}")
                    traceback.print_exc()
                    with csv_lock:
                        csv_writer.writerow([filename, 'ERROR', str(e), -1])

    mean_dist = np.mean(edit_distances)
    print(f'\nMean edit distance across {len(edit_distances)} images: {mean_dist:.4f}')
    print(f"Mismatch log saved to: {mismatches_path}")


def process_all_images():
    all_folders = os.listdir(image_directory)
    image_files = []
    for folder in all_folders:
        folder_path = os.path.join(image_directory, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(folder_path, filename))
    process_list(image_files)


def process_mismatches():
    mismatches = pd.read_csv(mismatches_path)
    if mismatches.empty:
        print("No mismatches found.")
        return
    process_list(list(mismatches['filename']))


#print(process_single_image("complete_dataset/images/56/G056_IMG023.jpg"))
process_all_images()
#process_mismatches()

"""
board = chess.Board(fen_string)
fen_svg = chess.svg.board(board=board)
output_filename = "chessboard_render.png"
cairosvg.svg2png(bytestring=fen_svg.encode('utf-8'), write_to=output_filename)
print(f"Chessboard image saved to {output_filename}")
"""

# complete_dataset/images/56/G056_IMG023.jpg
# 2r3q1/k1ppp1bp/p3p1b1/5n2/4N3/4NP2/P1PPPB1P/2R2Q
# r2q1rk1/ppp1bppp/2p1b3/3n4/2N5/2NP3P/PPPB1PP1/R2Q2KR
# 19

# complete_dataset/images/58/G033_IMG011.jpg,
# RP4pr/NP4p1/BP1B1npb/Q2p2pq/3n3k/RPN3pb/KP4p1/1P4pr
# r1bqkb1r/pppp1ppp/2n5/8/2Bpn3/5N2/PPP2PPP/RNBQ1RK1
# 45

# complete_dataset/images/33/G033_IMG011.jpg
# R3P1pr/1P4pn/BPN2pqb/Q2P1p2/K2Pp2k/BPN2npb/1P4p1/RP4pr
# rnb1kb1r/ppq2ppp/2pp1n2/P3p3/3PP3/2N2N2/1PP2PPP/R1BQKB1R
# 40