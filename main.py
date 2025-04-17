"""
`pip install opencv-python shapely pandas`

Run the file by doing `python3 main.py`.
This will run the image processing pipeline for the images in the `input.json` file and save the results in the `output.json` file.
"""

import cv2
import numpy as np
from shapely.geometry import Polygon
import pandas as pd

import os
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import json
import math

##==================================== Evaluation Helpers =========================================================#
# These are used for developing: only for evaluating which approach work better
# The dataset and labels are not used for the delivery

def get_piece_position(piece: str):
    """
    Get the position of the piece.
    """
    piece = piece.lower()
    row = ord(piece[1]) - ord("1")
    col = ord(piece[0]) - ord("a")
    return row, col

def get_corner_annotations(dataset):
    """
    Get the corner annotations.
    """
    corner_annotations = dataset["annotations"]["corners"]
    return corner_annotations

def get_piece_annotations(dataset):
    """
    Get the piece annotations.
    """
    piece_annotations = dataset["annotations"]["pieces"]
    return piece_annotations

def get_image_id_by_name(image_name, dataset):
    """
    Get image id by image name.
    """
    for image in dataset["images"]:
        if image["file_name"] == image_name:
            return image["id"]
    raise ValueError(
        f"""Could not find image id with name {image_name}.
        Likely, there has been an error related to the image name.""")

def get_annotations_by_image_name(image_name, dataset):
    """
    Get annotations by image name.
    """
    corner_annotations = get_corner_annotations(dataset)
    piece_annotations = get_piece_annotations(dataset)

    image_id = get_image_id_by_name(image_name, dataset)
    ans = {}
    ans["board"] = [[0] * 8 for _ in range(8)]
    ans["detected_pieces"] = []

    for annotation in corner_annotations:
        if annotation["image_id"] == image_id:
            ans["corners"] = annotation["corners"]

    for piece in piece_annotations:
        if piece["image_id"] == image_id:
            ans["detected_pieces"].append(
                {
                    "xmin": piece["bbox"][0],
                    "ymin": piece["bbox"][1],
                    "xmax": piece["bbox"][0] + piece["bbox"][2],
                    "ymax": piece["bbox"][1] + piece["bbox"][3],
                }
            )
            r, c = get_piece_position(piece["chessboard_position"])
            ans["board"][r][c] = 1
    return ans


def draw_bboxes(image, image_annotations, predictions: Optional[list] = None):
    """
    Show bounding boxes of the image.
    """
    for bbox in image_annotations["detected_pieces"]:
        cv2.rectangle(
            image,
            (int(bbox["xmin"]), int(bbox["ymin"])),
            (int(bbox["xmax"]), int(bbox["ymax"])),
            (0, 255, 0),
            2,
        )
    if predictions is not None:
        # Later, we could draw the predictions here
        pass
    return image

def draw_corners(image, image_annotations, predictions: Optional[list] = None):
    """
    Show corners of the image.
    """
    for corner in image_annotations["corners"].values():
        cv2.circle(image, (int(corner[0]), int(corner[1])), 20, (0, 255, 0), -1)
    if predictions is not None:
        # Later, we could draw the predictions here
        pass
    return image

def print_board(board):
    """
    Print the chessboard.
    """
    for row in board:
        print(row)
    print()


def from_annotation_to_output_json(image_name, annotation):
    """
    Convert annotation to json.
    """
    board = annotation["board"]
    num_pieces = sum([sum(row) for row in board])

    ans = {
        "image": image_name,
        "num_pieces": num_pieces,
        "board": board,
        "detected_pieces": annotation["bbox"],
    }
    if "corners" in annotation:
        ans["corners"] = annotation["corners"]

    return ans


def draw_annotations(image_name, image_path, dataset):
    image_annotations = get_annotations_by_image_name(image_name, dataset)

    image = cv2.imread(image_path)
    image = draw_bboxes(image, image_annotations, [])
    image = draw_corners(image, image_annotations, [])
    image = cv2.resize(image, (800, 800))

    return image

def evaluate_corners(true_corners, pred_corners, verbose: bool = False):
    corner_names = ["bottom_left", "bottom_right", "top_left", "top_right"]
    comparisons = {
        "bottom_left": ["bottom_left", "bottom_right", "top_right", "top_left"],
        "bottom_right": ["bottom_right", "top_right", "top_left", "bottom_left"],
        "top_right": ["top_right", "top_left", "bottom_left", "bottom_right"],
        "top_left": ["top_left", "bottom_left", "bottom_right", "top_right"],
    }

    min_corners_mse = float("inf")
    for i in range(4):
        corners_mse = 0
        pred_corners_copy = pred_corners.copy()

        for corner_name in corner_names:
            other_corner_name = comparisons[corner_name][i]
            pred_corners_copy[corner_name] = pred_corners[other_corner_name]

        for corner_name in corner_names:
            corners_mse += (
                true_corners[corner_name][0] - pred_corners_copy[corner_name][0]
            ) ** 2 + (
                true_corners[corner_name][1] - pred_corners_copy[corner_name][1]
            ) ** 2

        corners_mse = corners_mse / len(corner_names)
        min_corners_mse = min(min_corners_mse, corners_mse)
        if verbose:
            print(
                f"Corners MSE: {corners_mse:.0f}, orientation: {i}"
            )
    return min_corners_mse

def bbox_area(bbox: dict[str, float]):
    return (bbox["xmax"] - bbox["xmin"]) * (bbox["ymax"] - bbox["ymin"])

def bbox_intersection_area(box_a: dict[str, float], box_b: dict[str, float]):
    x_a = max(box_a["xmin"], box_b["xmin"])
    y_a = max(box_a["ymin"], box_b["ymin"])
    x_b = min(box_a["xmax"], box_b["xmax"])
    y_b = min(box_a["ymax"], box_b["ymax"])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    return inter_area

def iou(box_a: dict[str, float], box_b: dict[str, float]):
    inter_area = bbox_intersection_area(box_a, box_b)

    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area

def evaluate_bboxes(true_bboxes, pred_bboxes, verbose: bool = False):
    matches = []
    iou_threshold = 0.5
    used_pred_indices = set()

    for true_idx, true_box in enumerate(true_bboxes):
        best_iou = 0
        best_pred_idx = -1

        for pred_idx, pred_box in enumerate(pred_bboxes):
            if pred_idx in used_pred_indices:
                continue

            current_iou = iou(true_box, pred_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_pred_idx = pred_idx

        if best_iou >= iou_threshold:
            matches.append((true_idx, best_pred_idx, best_iou))
            used_pred_indices.add(best_pred_idx)

    tp = len(matches)
    fp = len(pred_bboxes) - tp
    fn = len(true_bboxes) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    if verbose:
        print("Bounding box statistics:")
        print("Matches (true_idx, pred_idx, iou):")
        for match in matches:
            print(f"- {match}")
        print()
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    return f1

def evaluate_board(true_board, pred_board, verbose: bool = False):
    if len(pred_board) != 8 or len(pred_board[0]) != 8:
        return 0
    tp = fp = fn = tn = 0
    for row in range(8):
        for col in range(8):
            t = true_board[row][col]
            p = pred_board[row][col]
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 1:
                fp += 1
            elif t == 1 and p == 0:
                fn += 1
            elif t == 0 and p == 0:
                tn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    if verbose:
        print("Board positions statistics:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    return f1

def evaluate_predictions(
    image_annotations,
    predictions,
    eval_corners: bool = True,
    eval_num_pieces: bool = True,
    eval_board: bool = True,
    eval_bboxes: bool = True,
    verbose: bool = True,
):
    true_board = image_annotations["board"]
    true_num_pieces = sum([sum(row) for row in true_board])
    true_bboxs = image_annotations["detected_pieces"]

    pred_board = predictions["board"]
    pred_num_pieces = sum([sum(row) for row in pred_board])
    pred_bboxs = predictions["detected_pieces"]

    num_pieces_diff = 0
    board_score = 0
    corners_mse = 0
    bbox_scores = 0
    if eval_num_pieces:
        # Eval number of pieces
        num_pieces_diff = abs(true_num_pieces - pred_num_pieces)
        if verbose:
            print(f"Num pieces diff: {num_pieces_diff}")

    if eval_bboxes:
        # Eval the bounding boxes
        bbox_scores = evaluate_bboxes(true_bboxs, pred_bboxs)
        if verbose:
            print(f"Bounding boxes scores: {bbox_scores}")

    if eval_board:
        # Eval the board
        board_score = evaluate_board(true_board, pred_board)
        if verbose:
            print(f"Board score: {board_score:.2f}")

    if eval_corners:
        # Eval the corners
        corners_mse = evaluate_corners(image_annotations["corners"], predictions["corners"])
        if verbose:
            print(f"Corners MSE: {corners_mse:.2f}")

    return {
        "num_pieces": num_pieces_diff,
        "board": board_score,
        "corners": corners_mse,
        "bboxes": bbox_scores,
    }

def show_all_annotations(dataset):
    image_list = list(sorted(os.listdir(os.path.join("data", "images"))))

    image_paths = {
        image_name: os.path.join("data", "images", image_name)
        for image_name in image_list
        if image_name.endswith(".jpg")
    }
    # don't need this for now
    # image_annotations = {
    #     image_name: get_annotations_by_image_name(image_name, dataset)
    #     for image_name in image_list
    # }

    output_dir = "annotated_images"
    os.makedirs(output_dir, exist_ok=True)
    for image_name in image_list:
        image_path = image_paths[image_name]
        # image_annotation_info = image_annotations[image_name]

        output_path = os.path.join(output_dir, image_name)
        result_image = draw_annotations(image_name, image_path, dataset)
        cv2.imwrite(output_path, result_image)
        print(f"Saved {output_path}")

def get_dataset():
    """
    Get the dataset.
    """
    with open("complete_dataset/annotations.json", "r") as f:
        dataset = json.load(f)
    return dataset

##==================================== Chessboard corner detection Helpers ====================================================##

def get_largest_contour(
    img: np.ndarray,
    image_path: str,
    image_name_prefix: str,
    canny_lower: int,
    canny_upper: int,
    min_distance_to_image_border: int,
    max_distance_to_merge_contours: int,
):
    canny = cv2.Canny(img, canny_lower, canny_upper, apertureSize=3, L2gradient=True)
    # finding contours on the canny edges
    contours, hierarchy = cv2.findContours(
        canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours: don't use any contours that are too close to the image border
    img_height, img_width = img.shape[:2]
    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if (
            x > min_distance_to_image_border
            and y > min_distance_to_image_border
            and x + w < img_width - min_distance_to_image_border
            and y + h < img_height - min_distance_to_image_border
        ):
            filtered_contours.append(cnt)

    contours = filtered_contours
    if not contours:
        print(f"No contours found in {image_path}")
        return

    # Get the largest contour by convex hull area
    largest_area = 0
    largest_contour = None
    largest_contour_without_ch = None
    for cnt in contours:
        contour_ch = cv2.convexHull(cnt)
        area = cv2.contourArea(contour_ch)
        if area > largest_area:
            largest_area = area
            largest_contour_without_ch = cnt
            largest_contour = contour_ch

    basic_contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(basic_contour_img, [largest_contour], 0, (0, 0, 255), 3)

    # Merge contours with largest contour if distance to the main convex hull is small
    # Go until no more contours can be merged
    changed = True
    contours_to_merge = []
    while changed:
        changed = False
        ncontours = []
        for cnt in contours:
            if cnt is largest_contour_without_ch:
                continue
            if len(cnt) < 3:  # not a polygon
                continue

            # contours are (n, 1, 2), take just the 2D points
            poly1 = Polygon(largest_contour[:, 0, :])
            poly2 = Polygon(cnt[:, 0, :])

            min_distance = poly1.distance(poly2)
            if min_distance <= max_distance_to_merge_contours:
                changed = True
                largest_contour = cv2.convexHull(np.vstack([largest_contour, cnt]))
                contours_to_merge.append(cnt)
            else:
                ncontours.append(cnt)
        contours = ncontours

    if largest_contour is None:
        print(f"No valid contour found in {image_path}")
        return

    # Draw the largest contour's convex hull: in red
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, [largest_contour], 0, (0, 0, 255), 3)

    # Draw the largest contour without considering its convex hull: in blue
    contour_no_ch_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_no_ch_img, [largest_contour_without_ch], 0, (255, 0, 0), 5)

    # Draw the merged contours (no convex hulls)
    cv2.drawContours(contour_no_ch_img, contours_to_merge, -1, (0, 255, 0), 3)

    # save all the images
    debug_images = {
        f"{image_name_prefix}_contour_no_ch_img": contour_no_ch_img,
        f"{image_name_prefix}_contour": contour_img,
        f"{image_name_prefix}_basic_contour_img": basic_contour_img,
        f"{image_name_prefix}_canny": canny,
    }

    # lambda im=im: im -> used to avoid problems with python lambda and enclosures
    ready_to_display_images = [(im_name, lambda im=im: im) for im_name, im in debug_images.items()]
    return {
        "images": debug_images,
        "display_images": ready_to_display_images,
        "largest_contour": largest_contour,
        "largest_contour_without_ch": largest_contour_without_ch,
        "contours_to_merge": contours_to_merge,
    }

def convex_hull_intersection(poly1, poly2):
    """
    Calculate the intersection of two polygons and return the convex hull of the intersection.
    If the intersection is empty, return the first polygon.
    """
    # Create polygons from the contours
    polygon1 = Polygon(poly1[:, 0, :])
    polygon2 = Polygon(poly2[:, 0, :])

    # Calculate the intersection
    intersection = polygon1.intersection(polygon2)

    # If there were issues with the intersection, return the first polygon
    if intersection.is_empty:
        return poly1

    if intersection.geom_type == "Polygon":
        x, y = intersection.exterior.coords.xy
        points = np.array(list(zip(x, y)), dtype=np.int32)
        return cv2.convexHull(points)

    return poly1

##==================================== Piece detection for each square using GrabCut =========================================##

def has_piece_grabcut(resized_img, original_corners, original_size, resized_size=(800, 800),
                      threshold_min=0.15, threshold_max=0.7, iterations=5, nr_pixels_below_piece=10):
    """
    Detects if a piece exists in a square using GrabCut on a resized image.

    Parameters:
        resized_img: the image that has been resized.
        original_corners: corners in original image coordinates.
        original_size: size of original image (width, height).
        resized_size: size of resized image (width, height).

    Returns:
        has_piece: True if a piece is detected.
        contour: largest contour in resized image coordinates.
    """
    # Scaling factors: original -> resized
    orig_w, orig_h = original_size
    resized_h, resized_w = resized_img.shape[:2]
    scale_x = resized_w / orig_w
    scale_y = resized_h / orig_h

    # Scale corners to match the resized image
    scaled_corners = [(int(x * scale_x), int(y * scale_y)) for (x, y) in original_corners]
    pts = np.array(scaled_corners, dtype=np.int32)

    # Crop ROI
    x, y, w, h = cv2.boundingRect(pts)
    square_img = resized_img[y:y+h, x:x+w]

    # Create GrabCut mask
    mask = np.full((h, w), cv2.GC_EVAL, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 6
    outer_radius = min(w, h) // 2.8
    cv2.circle(mask, (cx, cy), radius, cv2.GC_FGD, -1)

    # Create coordinate indices for the ROI.
    rows_idx, cols_idx = np.indices((h, w))
    # Compute the Euclidean distance from each pixel to the ROI center.
    distance = np.sqrt((cols_idx - cx)**2 + (rows_idx - cy)**2)
    
    # Initialize the mask: all pixels are set to probable background.
    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    # Inside the circle (distance <= piece_radius), mark as definite foreground.
    mask[distance <= radius] = cv2.GC_PR_FGD

    border_thickness = 10
    mask[:border_thickness, :] = cv2.GC_PR_BGD
    mask[-border_thickness:, :] = cv2.GC_PR_BGD
    mask[:, :border_thickness] = cv2.GC_PR_BGD
    mask[:, -border_thickness:] = cv2.GC_PR_BGD

    # Outside the circle—but only in the lower half (rows below the center) mark as definite background.
    mask[(distance > outer_radius) & (rows_idx > cy)] = cv2.GC_PR_BGD
    mask[-nr_pixels_below_piece:, :] = cv2.GC_PR_BGD

    hint_mask = mask.copy()
    # Run GrabCut
    bgModel = np.zeros((1, 65), dtype=np.float64)
    fgModel = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(square_img, mask, None, bgModel, fgModel, iterations, cv2.GC_INIT_WITH_MASK)

    result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    foreground_ratio = np.sum(result_mask) / float(w * h)
    has_piece = threshold_min < foreground_ratio and foreground_ratio < threshold_max

    # print(f"Detected piece: {has_piece} (foreground ratio: {foreground_ratio:.2f})")

    largest_contour = None
    if has_piece:
        contours, _ = cv2.findContours(result_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Offset contours to original image coordinates
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour += np.array([[x, y]])  # Broadcast the offset
            # Now, scale the contour's coordinates from the resized image back to the original image.
            largest_contour = largest_contour.astype(np.float32)
            largest_contour[:, 0, 0] /= scale_x  # x-coordinates
            largest_contour[:, 0, 1] /= scale_y  # y-coordinates
            largest_contour = largest_contour.astype(np.int32)

        # do it again making everything probable background except the piece ???

    color_map = {
        cv2.GC_BGD:    (0, 0, 255),   # Definite background: Red.
        cv2.GC_FGD:    (0, 255, 0),   # Definite foreground: Green.
        cv2.GC_PR_BGD: (255, 0, 0),   # Probable background: Blue.
        cv2.GC_PR_FGD: (0, 255, 255)  # Probable foreground: Yellow.
    }

    # Now, scale the local result_mask back to original coordinates.
    # Compute ROI location in original image coordinates.
    orig_roi_x = int(x / scale_x)
    orig_roi_y = int(y / scale_y)
    orig_roi_w = int(w / scale_x)
    orig_roi_h = int(h / scale_y)
    # Resize result_mask to the ROI's size in the original coordinate system.
    scaled_result_mask = cv2.resize(result_mask, (orig_roi_w, orig_roi_h), interpolation=cv2.INTER_NEAREST)
    
    # Create an overall mask (same size as the original image) and place the scaled ROI mask.
    overall_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    overall_mask[orig_roi_y:orig_roi_y+orig_roi_h, orig_roi_x:orig_roi_x+orig_roi_w] = scaled_result_mask

    # Create a color image for the hint mask.
    colored_hint_local = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        colored_hint_local[hint_mask == label] = color

    # Resize the colored hint mask back to original ROI size.
    colored_hint_resized = cv2.resize(colored_hint_local, (orig_roi_w, orig_roi_h), interpolation=cv2.INTER_NEAREST)
    
    # Create an overall colored hint mask (same size as original image) and place the resized ROI.
    hint_overall_mask_color = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    hint_overall_mask_color[orig_roi_y:orig_roi_y+orig_roi_h, orig_roi_x:orig_roi_x+orig_roi_w] = colored_hint_resized

    # Apply Otsu's binarization on the square image
    square_img_gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(square_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # This produced slightly worse results than just using grabcut
    """
    unique_vals, counts = np.unique(otsu_mask, return_counts=True)
    MIN_RATIO_TO_DETECT_PIECE = 0.2
    img_size = square_img_gray.shape[0] * square_img_gray.shape[1]
    min_pxs_to_detect_piece = MIN_RATIO_TO_DETECT_PIECE * img_size
    if counts[0] > min_pxs_to_detect_piece and counts[1] > min_pxs_to_detect_piece:
        has_piece = True
    else:
        has_piece = False
    """

    otsu_mask = cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels to match the final output

    # Resize the Otsu mask back to original ROI size
    otsu_resized = cv2.resize(otsu_mask, (orig_roi_w, orig_roi_h), interpolation=cv2.INTER_NEAREST)

    # Create an overall Otsu color mask for the entire original image size
    overall_otsu_color = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    overall_otsu_color[orig_roi_y:orig_roi_y+orig_roi_h, orig_roi_x:orig_roi_x+orig_roi_w] = otsu_resized

    # Apply K-means clustering to the square image (assumes a 2D color space)
    square_img_reshaped = square_img.reshape((-1, 3))  # Reshape to (N, 3) where N is the number of pixels
    square_img_reshaped = np.float32(square_img_reshaped)  # Convert to float32 for K-means


    return has_piece, largest_contour, hint_overall_mask_color, overall_mask * 255, foreground_ratio, overall_otsu_color

##==================================== Horse to detect the chessboard orientation ================================================##

def find_orientation(image):
    horse_path = "figures/horse.png"
    horse_img = cv2.imread(horse_path, cv2.IMREAD_GRAYSCALE)
    
    if horse_img is None:
        print("Could not load the horse template image")
        return None

    height, width = image.shape
    corner_size = min(width, height) // 4
    
    orientations = {
        "top_left": (image[:corner_size, :corner_size], cv2.rotate(horse_img, cv2.ROTATE_90_CLOCKWISE)),
        "top_right": (image[:corner_size, width-corner_size:], cv2.rotate(horse_img, cv2.ROTATE_180)),
        "bottom_left": (image[height-corner_size:, :corner_size], horse_img),
        "bottom_right": (image[height-corner_size:, width-corner_size:], cv2.rotate(horse_img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    }
    
    best_score = -1
    best_match_loc = None
    best_rotation = None
    
    for corner_name, (corner_img, horse_template) in orientations.items():
        target_size = corner_size // 4
        resized_template = cv2.resize(horse_template, (target_size, target_size))
        
        if corner_img.shape[0] < resized_template.shape[0] or corner_img.shape[1] < resized_template.shape[1]:
            continue
            
        result = cv2.matchTemplate(corner_img, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            
            if corner_name == "top_left":
                match_x, match_y = max_loc
                best_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif corner_name == "top_right":
                match_x, match_y = width - corner_size + max_loc[0], max_loc[1]
                best_rotation = cv2.ROTATE_180
            elif corner_name == "bottom_left":
                match_x, match_y = max_loc[0], height - corner_size + max_loc[1]
                best_rotation = None
            elif corner_name == "bottom_right":
                match_x, match_y = width - corner_size + max_loc[0], height - corner_size + max_loc[1]
                best_rotation = cv2.ROTATE_90_CLOCKWISE
                
            best_match_loc = (match_x, match_y)

    return best_rotation, best_match_loc

##==================================== Given chessboard orientation, rotate board ===============================================##
def _rotate_90(board):
    """
    Rotates the board 90 degrees clock-wise.

    If the board IMAGE needs to be rotated 90 degrees counterclock-wise,
    then the board GRID should be rotated 90 degrees clock-wise instead.
    """
    return [list(reversed(col)) for col in zip(*board)]

def adjust_board(board, rotation):
    """
    Adjusts the orientation of a board representation based on a given rotation.

    Good test images:
    - G000_IMG062.jpg
    - G000_IMG087.jpg
    - G006_IMG119.jpg
    - G033_IMG043.jpg

    Parameters:
    ----------
    board : list
        A 2D list representing the board, where each sublist is a row.

    rotation : cv2.RotateFlag or None
        The rotation that the board image would be applied to make the horse be bottom-left. Can be one of:
        - cv2.ROTATE_180
        - cv2.ROTATE_90_CLOCKWISE
        - cv2.ROTATE_90_COUNTERCLOCKWISE
        - None (no rotation)
        For instance, if the horse was at the top left, then the rotation would be cv2.ROTATE_90_COUNTERCLOCKWISE

    Returns:
    -------
    list
        The adjusted board after applying the appropriate transformation.
    """

    # Need to initially adjust board since we represent it differently than in the dataset label
    adjusted_board = list(reversed(board))    # reverse board (columns)

    iters = 0
    if rotation == cv2.ROTATE_180:
        # Flip original board vertically and horizontally
        iters = 2   # or instead rotate twice
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        iters = 1
    elif rotation == cv2.ROTATE_90_CLOCKWISE:
        iters = 3   # 90 cw = 3 * 90 ccw

    for _ in range(iters):
        adjusted_board = _rotate_90(adjusted_board)
    return adjusted_board

##==================================== Square and Piece Detection Helpers ====================================================##

def filter_and_rectify_hough_lines(lines, image_shape, angle_threshold=10, distance_threshold=20):

    vertical_candidates = []
    horizontal_candidates = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        theta = theta % 180
        
        if theta < angle_threshold or theta > (180 - angle_threshold):
            y_mean = (y1 + y2) // 2
            horizontal_candidates.append(y_mean)
        elif abs(theta - 90) < angle_threshold:
            x_mean = (x1 + x2) // 2
            vertical_candidates.append(x_mean)

    # Cluster similar horizontal lines by their y coordinate
    horizontals = []
    horizontal_candidates.sort()
    for y in horizontal_candidates:
        if not horizontals or abs(y - horizontals[-1]) > distance_threshold:
            horizontals.append(y)
        else:
            horizontals[-1] = (horizontals[-1] + y) // 2

    # Cluster similar vertical lines by their x coordinate
    verticals = []
    vertical_candidates.sort()
    for x in vertical_candidates:
        if not verticals or abs(x - verticals[-1]) > distance_threshold:
            verticals.append(x)
        else:
            verticals[-1] = (verticals[-1] + x) // 2

    return verticals, horizontals

def identify_and_add_missing_lines(verticals, horizontals, image_shape, max_gap_ratio=1.8):
    verticals.sort()
    horizontals.sort()
    
    new_verticals = verticals.copy()
    if len(verticals) >= 2:
        gaps = [verticals[i+1] - verticals[i] for i in range(len(verticals)-1)]
        median_gap = sorted(gaps)[len(gaps)//2]  # Using median is more robust than mean
        
        for i in range(len(verticals)-1):
            current_gap = verticals[i+1] - verticals[i]
            if current_gap > max_gap_ratio * median_gap:
                # Calculate how many lines are missing
                n_missing = round(current_gap / median_gap) - 1
                for j in range(1, n_missing + 1):
                    # Add estimated line position
                    new_x = verticals[i] + j * (current_gap / (n_missing + 1))
                    new_verticals.append(int(new_x))
    
    new_horizontals = horizontals.copy()
    if len(horizontals) >= 2:
        gaps = [horizontals[i+1] - horizontals[i] for i in range(len(horizontals)-1)]
        median_gap = sorted(gaps)[len(gaps)//2]
        
        for i in range(len(horizontals)-1):
            current_gap = horizontals[i+1] - horizontals[i]
            if current_gap > max_gap_ratio * median_gap:
                n_missing = round(current_gap / median_gap) - 1
                for j in range(1, n_missing + 1):
                    new_y = horizontals[i] + j * (current_gap / (n_missing + 1))
                    new_horizontals.append(int(new_y))
    
    new_verticals.sort()
    new_horizontals.sort()
    
    return new_verticals, new_horizontals

def compute_intersections(verticals, horizontals):
    intersections = []
    for x in verticals:
        for y in horizontals:
            intersections.append((x, y))
    return intersections

def filter_intersections_by_distance(intersections, center):
    """
    Choose the 81 intersections that are closest to the center of the board,
    ensuring each point is at least 250 pixels away from any other selected point.
    """
    x_center, y_center = center
    distances = []
    for point in intersections:
        x, y = point
        distance = math.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        distances.append((distance, point))

    distances.sort(key=lambda x: x[0])
    
    filtered_intersections = [distances[0][1]]
    
    for _, point in distances[1:]:
        valid_point = True
        for selected_point in filtered_intersections:
            x1, y1 = point
            x2, y2 = selected_point
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < 250:
                valid_point = False
                break
        
        if valid_point:
            filtered_intersections.append(point)
            
        if len(filtered_intersections) == 81:
            break
    
    square_side = int(math.sqrt(len(filtered_intersections)))
    
    if len(filtered_intersections) < 81:
        print(f"Warning: Only found {len(filtered_intersections)} valid intersections with minimum distance of 250 pixels")
    
    return filtered_intersections, square_side

def has_piece(image, square_vertices) -> bool:  # TODO: Improve this algorithm
    # Create a mask for the square region
    mask = np.zeros(image.shape, dtype=np.uint8)
    pts = np.array([square_vertices], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    
    # Extract the square region using the mask
    square_region = cv2.bitwise_and(image, image, mask=mask)
    
    # Get bounding box of the square for cropping
    x_coords = [p[0] for p in square_vertices]
    y_coords = [p[1] for p in square_vertices]
    x_min, x_max = max(0, min(x_coords)), min(image.shape[1], max(x_coords))
    y_min, y_max = max(0, min(y_coords)), min(image.shape[0], max(y_coords))
    
    # Crop to bounding box
    cropped = square_region[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return False
    
    # Apply thresholding to separate pieces from background
    _, thresh = cv2.threshold(cropped, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Calculate the area of the square
    square_area = (x_max - x_min) * (y_max - y_min)
    
    # Check for sufficiently large contours that could be pieces
    for contour in contours:
        area = cv2.contourArea(contour)
        # A piece should occupy a significant portion of the square but not too much
        area_ratio = area / square_area
        
        if 0.05 < area_ratio < 0.8:
            # Calculate circularity/roundness of the contour
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Ellipses/circles have circularity closer to 1
                # But incomplete ellipses will have lower values
                if 0.2 < circularity < 0.9:
                    return True
    
    return False


##================================= Helpers to get piece bounding boxes ============================================##

def get_white_pieces_mask(warped_hsv):
    whl, whu = 15, 40    # hue ~ yellow, saturation > X, value > Y
    wsl, wsu = 25, 255
    wvl, wvu = 100, 255

    # white_lower = np.array([hl, sl, vl])
    # white_upper = np.array([hu, su, vu])
    # white_mask = cv2.inRange(warped_hsv, white_lower, white_upper)

    # Instead of splitting everything at once, do it individually
    # Split channels
    h, s, v = cv2.split(warped_hsv)

    # Create individual binary masks
    # For debugging purposes, using an "and" manually helps (since we can inspect each part manually)
    white_h_mask = (h >= whl) & (h <= whu)
    white_s_mask = (s >= wsl) & (s <= wsu)
    white_v_mask = (v >= wvl) & (v <= wvu)

    # Instead of threshold value, can do it like this:
    # gs_img = cv2.cvtColor(warped_img_color_blurred, cv2.COLOR_BGR2GRAY)
    # _, gs_v_th = cv2.threshold(gs_img, 100, 255, cv2.THRESH_BINARY)
    # v_mask = gs_v_th

    # Combine them with AND
    white_hsv_mask = white_h_mask & white_s_mask & white_v_mask
    white_mask = (white_hsv_mask.astype(np.uint8)) * 255

    final_white_mask = cv2.erode(white_mask, None, iterations=5)
    final_white_mask = cv2.dilate(final_white_mask, None, iterations=5)

    # Debugging purposes
    white_h_mask = np.uint8(white_h_mask) * 255
    white_s_mask = np.uint8(white_s_mask) * 255
    white_v_mask = np.uint8(white_v_mask) * 255
    return final_white_mask, white_mask, (white_h_mask, white_s_mask, white_v_mask)

def find_contours_in_mask(
    mask_img,
    min_piece_contour_area: int,
    max_piece_contour_area: int,
    min_piece_bbox_width: int,
    min_piece_bbox_height: int,
):
    contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    contour_areas = [cv2.contourArea(c) for c in contours]
    contour_ch_areas = [cv2.contourArea(cv2.convexHull(c)) for c in contours]
    contours = [
        c
        for c, a, ch_a in zip(contours, contour_areas, contour_ch_areas)
        if min_piece_contour_area <= a and ch_a <= max_piece_contour_area
    ]

    # (x, y, w, h) = cv2.boundingRect(c)
    bboxes = [cv2.boundingRect(c) for c in contours]
    # Filter contours by bbox width and height
    contours = [
        c for c, bbox in zip(contours, bboxes)
        if bbox[2] > min_piece_bbox_width and bbox[3] > min_piece_bbox_height
    ]
    return contours

##================================= Converting contours/bboxes back to original space ============================================##

def draw_bboxes(img, bboxes, color=(0, 255, 0)):
    for xmin, ymin, xmax, ymax in bboxes:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 10)

def get_bboxes(contours):
    bboxes = map(cv2.boundingRect, contours)
    bboxes = [
        (x, y, x + w, y + h)
        for x, y, w, h in bboxes
    ]
    return bboxes

def bboxes_to_dicts(bboxes):
    ans = []
    for xmin, ymin, xmax, ymax in bboxes:
        ans.append({
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })
    return ans

"""
Cannot convert bounding boxes to original
Need to convert contours
Otherwise bounding boxes conversion will lead to high imprecisions, because of perspective transform
That happens because we need to do the bounding boxes again in the original image (after converting the bounding boxes)
"""
def convert_contours_to_original(original_transform, contours):
    # Calculate the inverse perspective transform
    inverse_transform = np.linalg.inv(original_transform)

    # Prepare converted contours list
    original_contours = []

    for contour in contours:
        # Convert contour to the shape (N, 1, 2) and type float32
        contour_float = contour.astype(np.float32)

        # Apply inverse perspective transform
        original_contour = cv2.perspectiveTransform(contour_float, inverse_transform)

        # Convert back to int and append
        original_contours.append(original_contour.astype(np.int32))

    return original_contours

##====================================== MAIN IMAGE PROCESSING FUNCTION =========================================================##

def process_image(
    image_path,
    output_dir: Optional[str] = None,
    output_config: Optional[dict] = None,
    is_delivery: bool = False,
    default_show_image: bool = False
):
    # Hyperparameters

    # For corner detection
    THRESHOLD_MAXVAL: int = 255
    THRESHOLD_THRESH: int = 200
    THRESHOLD_SATURATION_MAX: int = 50
    CANNY_LOWER: int = 100
    CANNY_UPPER: int = 250
    MAX_DISTANCE_TO_MERGE_CONTOURS: int = 5
    MIN_DISTANCE_TO_IMAGE_BORDER: int = 10
    FILTER_BY_SATURATION: bool = False

    # For bbox detection
    MIN_PIECE_CONTOUR_AREA: int = int(2e4)
    MAX_PIECE_CONTOUR_AREA: int = int(5e5)
    MIN_PIECE_BBOX_WIDTH: int = 30
    MIN_PIECE_BBOX_HEIGHT: int = 30

    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"Failed to load image: {image_path}")
        return

    if FILTER_BY_SATURATION:
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

        s = hsv[:, :, 1]    # Keep low-saturation areas (so threshold is inverted)
        _, s_thresh = cv2.threshold(s, THRESHOLD_SATURATION_MAX, THRESHOLD_MAXVAL, cv2.THRESH_BINARY_INV)

        # Convert to grayscale
        img_to_blur = cv2.bitwise_and(img, img, mask=s_thresh)
    else:
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_to_blur = img.copy()

    # Apply a Gaussian blur - To eliminate the noise in segmentation
    blur_img = cv2.GaussianBlur(img_to_blur, (11, 11), 0)

    # Apply global binary threshold - Board segmentation (by intensity)
    _, th_global = cv2.threshold(
        blur_img, THRESHOLD_THRESH, THRESHOLD_MAXVAL, cv2.THRESH_BINARY
    )

    base_img_contour_results = get_largest_contour(
        img,
        image_path,
        "base_img",
        CANNY_LOWER,
        CANNY_UPPER,
        MIN_DISTANCE_TO_IMAGE_BORDER,
        MAX_DISTANCE_TO_MERGE_CONTOURS,
    )
    threshold_contour_results = get_largest_contour(
        th_global,
        image_path,
        "threshold_img",
        CANNY_LOWER,
        CANNY_UPPER,
        MIN_DISTANCE_TO_IMAGE_BORDER,
        MAX_DISTANCE_TO_MERGE_CONTOURS,
    )

    largest_contour_base_img = base_img_contour_results["largest_contour"]
    largest_contour_threshold = threshold_contour_results["largest_contour"]

    largest_contour = convex_hull_intersection(
        largest_contour_base_img, largest_contour_threshold
    )

    base_img_contour_images = base_img_contour_results["display_images"]
    threshold_contour_images = threshold_contour_results["display_images"]

    # --- Corner Detection ---
    # Approximate the contour to a polygon with four points
    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = 0.01 * perimeter  # Initial approximation parameter (1% of perimeter)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Adjust epsilon until we get exactly 4 points
    max_attempts = 10
    attempt = 0
    while len(approx) != 4 and attempt < max_attempts:
        epsilon *= 1.2  # Increase epsilon by 20% each iteration
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        attempt += 1

    if len(approx) != 4:
        print(
            f"Could not approximate to four points for {image_path} after {max_attempts} attempts"
        )
        return

    # Extract the four points
    points = [pt[0] for pt in approx]

    # Order points: top-left, top-right, bottom-left, bottom-right
    points.sort(key=lambda p: p[1])  # Sort by y-coordinate
    top_points = points[:2]
    bottom_points = points[2:]
    top_points.sort(key=lambda p: p[0])  # Sort top points by x-coordinate
    bottom_points.sort(key=lambda p: p[0])  # Sort bottom points by x-coordinate
    top_left, top_right = top_points
    bottom_left, bottom_right = bottom_points

    # Draw the points on a copy of the original image
    points_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt in [top_left, top_right, bottom_left, bottom_right]:
        cv2.circle(points_img, tuple(map(int, pt)), 25, (0, 0, 255), -1)
    cv2.drawContours(points_img, [largest_contour], 0, (0, 255, 0), 3)

    # Define comparison points (destination corners)
    top_left_comp = (0, 0)
    bottom_left_comp = (0, img.shape[0])
    top_right_comp = (img.shape[1], 0)
    bottom_right_comp = (img.shape[1], img.shape[0])

    # Create the warp matrix based on the points and the destination points
    warp_matrix = cv2.getPerspectiveTransform(
        np.float32([top_left, bottom_left, top_right, bottom_right]),
        np.float32(
            [top_left_comp, bottom_left_comp, top_right_comp, bottom_right_comp]
        ),
    )

    # Apply the warp matrix to the image    
    warped_gray_img = cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]))
    warped_color_img = cv2.warpPerspective(img_color, warp_matrix, (img.shape[1], img.shape[0]))

    # Board orientation
    image_rotation, horse_location = find_orientation(warped_gray_img)
    if image_rotation is not None:
        rotated_img = cv2.rotate(warped_gray_img, image_rotation)
    else:
        rotated_img = warped_gray_img.copy()
    
    horse_img = cv2.cvtColor(warped_gray_img, cv2.COLOR_GRAY2BGR)
    cv2.circle(horse_img, horse_location, 50, (0, 0, 255), -1)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(warped_gray_img)

    # Gaussian Blur
    blurred_warped = cv2.GaussianBlur(clahe_img, (7, 7), 0)

    # Canny Edge Detection
    canny = cv2.Canny(blurred_warped, 50, 150)

    # Dilation
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(
        dilated, 1, np.pi / 180, 500, minLineLength=1700, maxLineGap=400
    )

    # Draw lines on the original image
    hough_lines_img = cv2.cvtColor(warped_gray_img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print(f"No lines detected in {image_path}")
        return

    if lines is not None:
        verticals, horizontals = filter_and_rectify_hough_lines(
            lines, img.shape, angle_threshold=10, distance_threshold=20
        )
        
        verticals, horizontals = identify_and_add_missing_lines(verticals, horizontals, img.shape)
        
        rectified_verticals = [(x, 0, x, img.shape[0]) for x in verticals]
        rectified_horizontals = [(0, y, img.shape[1], y) for y in horizontals]
        
        hough_lines_rectified_img = cv2.cvtColor(warped_gray_img, cv2.COLOR_GRAY2BGR)
        filtered_intersections_img = cv2.cvtColor(warped_gray_img, cv2.COLOR_GRAY2BGR)
        
        # vertical lines
        for x, y0, x2, y2 in rectified_verticals:
            cv2.line(hough_lines_rectified_img, (x, y0), (x2, y2), (255, 200, 0), 10)
            
        # horizontal lines
        for x0, y, x2, y2 in rectified_horizontals:
            cv2.line(hough_lines_rectified_img, (x0, y), (x2, y2), (255, 200, 0), 10)
        
        intersections = compute_intersections(verticals, horizontals)
        for point in intersections:
            cv2.circle(hough_lines_rectified_img, point, 25, (0, 0, 255), -1)
        
        x_center, y_center = warped_gray_img.shape[1] // 2, warped_gray_img.shape[0] // 2

        filtered_intersections, square_side = filter_intersections_by_distance(intersections, (x_center, y_center))
        for point in filtered_intersections:
            cv2.circle(filtered_intersections_img, point, 25, (0, 255, 0), -1)

    else:
        print(f"No lines detected in {image_path}")

    filtered_intersections.sort(key=lambda p: (p[1], p[0]))
    
    if len(filtered_intersections) >= 81:
        rows = cols = 8
        step = 9
    else:
        grid_side = int(math.sqrt(len(filtered_intersections))) - 1
        rows = cols = grid_side if grid_side > 0 else 0
        step = grid_side + 1
        print(f"Using a {rows}x{cols} grid with {len(filtered_intersections)} intersections")

    # Avoid direct errors (e.g. did not detect complete board)
    board = [[0] * 8 for _ in range(8)]
    # board = [[0] * cols for _ in range(rows)] if rows > 0 and cols > 0 else [[0] * 8]

    pieces_img = warped_color_img.copy()
    pieces_img_gc = warped_color_img.copy()
    warped_img_color_blurred = cv2.GaussianBlur(warped_color_img, (5, 5), 0)
    gc_resized_size = (800, 800)
    warped_img_color_blurred_resized = cv2.resize(warped_img_color_blurred, gc_resized_size)

    final_mask = np.zeros(pieces_img.shape[:2], dtype=np.uint8)
    foreground_ratios = np.zeros(pieces_img.shape[:2], dtype=np.uint8)
    gc_hint_mask_color = np.zeros_like(warped_color_img, dtype=np.uint8)
    otsu_mask = np.zeros_like(warped_color_img, dtype=np.uint8)

    has_piece_contours = []
    for i in range(rows):
        for j in range(cols):
            if (i+1) * step + j + 1 < len(filtered_intersections):
                square_corners = [
                    filtered_intersections[i * step + j],
                    filtered_intersections[i * step + j + 1],
                    filtered_intersections[(i + 1) * step + j + 1],
                    filtered_intersections[(i + 1) * step + j],
                ]
                grabcut_has_piece, contour, hint_local_mask, local_mask, fg_ratio, otsu_result = has_piece_grabcut(
                    warped_img_color_blurred_resized,
                    square_corners,
                    original_size=img.shape,
                    resized_size=gc_resized_size,
                )

                pts = np.array(square_corners, dtype=np.int32)
                rect = cv2.boundingRect(pts)
                center_x = rect[0] + rect[2] // 2
                center_y = rect[1] + rect[3] // 2
                percentage_text = f"{fg_ratio*100:.0f}"
                cv2.putText(foreground_ratios, percentage_text, (center_x - 100, center_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 2)

                final_mask = cv2.bitwise_or(final_mask, local_mask)
                gc_hint_mask_color = cv2.bitwise_or(gc_hint_mask_color, hint_local_mask)
                otsu_mask = cv2.bitwise_or(otsu_mask, otsu_result)
                if grabcut_has_piece:
                    board[i][j] = 1
                    has_piece_contours.append(contour)

                    # Get the bounding rectangle from the contour for drawing purposes.
                    contour_bbox = cv2.boundingRect(contour)
                    # Draw the bounding rectangle on the pieces image.
                    x, y, w, h = contour_bbox
                    cv2.polylines(pieces_img_gc, [np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ])], True, (0, 0, 255), 10)

                    cv2.polylines(pieces_img_gc, [np.array(square_corners)], True, (0, 255, 0), 10)
                    cv2.polylines(foreground_ratios, [np.array(square_corners)], True, 255, 10)

 

                # This produced worse results in comparison to grabcut
                # if has_piece(warped_gray_img, square_corners):
                #     board[i][j] = 1
                #     cv2.polylines(pieces_img, [np.array(square_corners)], True, (0, 0, 255), 10)

    ##===================== Detect pieces bounding boxes =========================##

    warped_hsv = cv2.cvtColor(warped_img_color_blurred, cv2.COLOR_BGR2HSV)
    warped_lab = cv2.cvtColor(warped_img_color_blurred, cv2.COLOR_BGR2LAB)

    white_mask, final_white_mask, (white_h_mask, white_s_mask, white_v_mask) = get_white_pieces_mask(warped_hsv)

    # Instead of splitting everything at once, do it individually
    # Split channels
    h, s, v = cv2.split(warped_hsv)
    l, _, _ = cv2.split(warped_lab)

    # Apply CLAHE
    # h = clahe.apply(h)
    s = clahe.apply(s)
    v = clahe.apply(v)
    l = clahe.apply(l)
    
    hsv_clahed = cv2.merge((h, s, v))
    bgr_clahed = cv2.cvtColor(hsv_clahed, cv2.COLOR_HSV2BGR)

    bhl, bhu = 0, 255
    bsl, bsu = 0, 80
    bvl, bvu = 0, 80
    # black_lower = np.array([0, 0, 0])
    # black_upper = np.array([180, 50, 80])  # saturation < 100, value < 80
    # black_lower = np.array([bhl, bsl, bvl])
    # black_upper = np.array([bhu, bsu, bvu])
    # black_mask = cv2.inRange(warped_hsv, black_lower, black_upper)

    black_h_mask = (h >= bhl) & (h <= bhu)
    black_s_mask = (s >= bsl) & (s <= bsu)
    black_v_mask = (l >= bvl) & (l <= bvu)
    black_hsv_mask = black_h_mask & black_s_mask & black_v_mask
    black_mask = (black_hsv_mask.astype(np.uint8)) * 255

    # Debugging purposes
    black_h_mask = np.uint8(black_h_mask) * 255
    black_s_mask = np.uint8(black_s_mask) * 255
    black_v_mask = np.uint8(black_v_mask) * 255

    morphological_op_black_mask = cv2.dilate(black_mask, None, iterations=15)
    # morphological_op_black_mask = cv2.erode(morphological_op_black_mask, None, iterations=12)

    # morphological_op_black_mask = cv2.morphologyEx(morphological_op_black_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    # morphological_op_black_mask = cv2.erode(morphological_op_black_mask, None, iterations=10)

    # black_mask = cv2.erode(black_mask, None, iterations=2)
    # black_mask = cv2.dilate(black_mask, None, iterations=2)
    # # make it more blob-like (using morphology)
    # kernel = np.ones((5, 5), np.uint8)
    # black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=20)

    dist_transform = cv2.distanceTransform(morphological_op_black_mask, cv2.DIST_L2, 5)
    _, dist_transform_sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)

    dist_transform_sure_fg = np.uint8(dist_transform_sure_fg)
    sure_bg = cv2.dilate(morphological_op_black_mask, None, iterations=5)
    unknown = cv2.subtract(sure_bg, dist_transform_sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(dist_transform_sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(warped_img_color_blurred, markers)
    markers[markers > 1] = 255
    markers[markers == 1] = 0
    markers = np.uint8(markers)

    sure_bg = np.uint8(sure_bg)
    dist_transform_sure_fg = np.uint8(dist_transform_sure_fg)

    piece_contours_img = warped_color_img.copy()
    bboxes_img = warped_color_img.copy()
    original_bboxes_image = img_color.copy()

    # Find contours in white mask
    white_contours = find_contours_in_mask(
        final_white_mask,
        MIN_PIECE_CONTOUR_AREA,
        MAX_PIECE_CONTOUR_AREA,
        MIN_PIECE_BBOX_WIDTH,
        MIN_PIECE_BBOX_HEIGHT,
    )
    cv2.drawContours(piece_contours_img, white_contours, -1, (255, 0, 0), 5)

    final_black_mask = np.uint8(markers)
    black_contours = find_contours_in_mask(
        final_black_mask,
        MIN_PIECE_CONTOUR_AREA,
        MAX_PIECE_CONTOUR_AREA,
        MIN_PIECE_BBOX_WIDTH,
        MIN_PIECE_BBOX_HEIGHT,
    )

    white_bboxes = get_bboxes(white_contours)
    draw_bboxes(bboxes_img, white_bboxes)

    black_bboxes = get_bboxes(black_contours)
    draw_bboxes(bboxes_img, black_bboxes, (0, 0, 255))

    all_contours = white_contours.copy()
    INCLUDE_BLACK_MASK_CONTOURS = True
    if INCLUDE_BLACK_MASK_CONTOURS:
        all_contours.extend(black_contours)

    ##===================================================================================##
    INCLUDE_ADDITIONAL_BBOXES = False
    if INCLUDE_ADDITIONAL_BBOXES:
        # Hyperparameters
        MAX_IOU_TO_INCLUDE = 0.5
        MAX_INTERSECTION_AREA_RATIO_TO_INCLUDE = 0.1    # much stronger filtering than the IOU
        TOO_MANY_CONTOURS = 32

        if len(has_piece_contours) > TOO_MANY_CONTOURS:
            print(f"""Warning: Too many contours ({len(has_piece_contours)}) found using `has_piece` in {image_path}.
                  Ignored these contours to avoid too many FP""")
        else:
            has_piece_bboxes = get_bboxes(has_piece_contours)
            has_piece_bboxes_dicts = bboxes_to_dicts(has_piece_bboxes)
            white_bboxes_dicts = bboxes_to_dicts(white_bboxes)

            additional_contours_from_has_pieces = []

            for has_piece_bbox_dict, has_piece_contour in zip(has_piece_bboxes_dicts, has_piece_contours):
                has_piece_bbox_area = bbox_area(has_piece_bbox_dict)
                include = True
                
                for white_bbox_dict in white_bboxes_dicts:
                    inter_area = bbox_intersection_area(white_bbox_dict, has_piece_bbox_dict)
                    white_bbox_area = bbox_area(white_bbox_dict)

                    if (
                        iou(white_bbox_dict, has_piece_bbox_dict) > MAX_IOU_TO_INCLUDE
                        or inter_area > MAX_INTERSECTION_AREA_RATIO_TO_INCLUDE * white_bbox_area
                        or inter_area > MAX_INTERSECTION_AREA_RATIO_TO_INCLUDE * has_piece_bbox_area
                    ):
                        include = False
                        break
                if include:
                    additional_contours_from_has_pieces.append(has_piece_contour)

            all_contours.extend(additional_contours_from_has_pieces)
            additional_bboxes = get_bboxes(additional_contours_from_has_pieces)
            draw_bboxes(bboxes_img, additional_bboxes, (0, 0, 255))

    original_contours = convert_contours_to_original(warp_matrix, all_contours)
    original_bboxes = get_bboxes(original_contours)
    draw_bboxes(original_bboxes_image, original_bboxes)
    bboxes_ans = bboxes_to_dicts(original_bboxes)

    if output_dir is not None:
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        image_folder = os.path.join(output_dir, base_filename)
        os.makedirs(image_folder, exist_ok=True)

        output_handlers = [
            ("original", lambda: img),
            ("corners", lambda: points_img),
            ("threshold", lambda: th_global),
            ("warped", lambda: warped_gray_img),
            *base_img_contour_images,
            *threshold_contour_images,
            ('clahe', lambda: clahe_img),
            ('blurred_warped', lambda: blurred_warped),
            ('canny_edges', lambda: canny),
            ('dilated', lambda: dilated),
            ('hough_lines', lambda: hough_lines_img),
            ('hough_lines_rectified', lambda: hough_lines_rectified_img),
            ('filtered_intersections', lambda: filtered_intersections_img),
            ('pieces_gc', lambda: pieces_img_gc),
            ('warped_color', lambda: warped_color_img),
            ('grabcut_mask', lambda: final_mask),
            ('grabcut_fg_ratios', lambda: foreground_ratios),
            ('grabcut_hint_mask', lambda: gc_hint_mask_color),
            ('black_pieces_mask', lambda: black_mask),
            ('black_pieces_mask_morph', lambda: morphological_op_black_mask),
            ('white_pieces_mask', lambda: white_mask),
            ('white_pieces_mask_morph', lambda: final_white_mask),
            ('white_hue_mask', lambda: white_h_mask),
            ('white_sat_mask', lambda: white_s_mask),
            ('white_val_mask', lambda: white_v_mask),
            ('black_hue_mask', lambda: black_h_mask),
            ('black_sat_mask', lambda: black_s_mask),
            ('black_val_mask', lambda: black_v_mask),
            ('bboxes', lambda: bboxes_img),
            ('piece_contours', lambda: piece_contours_img),
            ('otsu_mask', lambda: otsu_mask),
            ('bboxes_orig', lambda: original_bboxes_image),
            ('pieces', lambda: pieces_img),
            ('horse', lambda: horse_img),
            ('rotated', lambda: rotated_img),
            ('bgr_clahed', lambda: bgr_clahed),
            ('dist_transform_sure_fg', lambda: dist_transform_sure_fg),
            ('sure_bg', lambda: sure_bg),
            ('unknown', lambda: unknown),
            ('markers', lambda: markers),
        ]

        for output_type, get_image in output_handlers:
            # If not within the configuration, show it if default is True
            if output_config.get(output_type, default_show_image):
                image = get_image()
                cv2.imwrite(
                    os.path.join(image_folder, f"{base_filename}_{output_type}.jpg"),
                    image,
                )

    adjusted_board = adjust_board(board, image_rotation)

    predictions = {
        "image": image_path,
        "corners": {
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
            "top_left": top_left,
            "top_right": top_right,
        },
        "board": adjusted_board,
        "detected_pieces": bboxes_ans,
        "num_pieces": sum([sum(row) for row in board]),
    }
    print(f"Processed {image_path}")
    return predictions


def process_all_images(
    output_dir,
    output_config,
    eval_predictions: bool = True,
    show_all_images: bool = False,
    max_workers: int = None,
):
    images_dir = "./data/images"
    output = []
    
    # Get a list of all image files
    image_files = [
        filename for filename in sorted(os.listdir(images_dir))
        if filename.endswith((".jpg", ".jpeg", ".png"))
    ]
    
    if not image_files:
        print("No image files found in directory.")
        return
    
    # Multi-threading setup
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    results_queue = Queue()
    lock = threading.Lock()
    
    if eval_predictions:
        dataset = get_dataset()
        image_evaluations = []

    print(f"Processing {len(image_files)} images with {max_workers} threads...")
    
    # Define a worker function to process a single image
    def process_image_worker(filename):
        image_path = os.path.join(images_dir, filename)
        try:
            predictions = process_image(image_path, output_dir, output_config, is_delivery=False)
            if predictions:
                results_queue.put({
                    "image": image_path,
                    "num_pieces": predictions['num_pieces'],
                    "board": predictions['board'],
                    "detected_pieces": predictions['detected_pieces'],
                    "filename": filename,
                })

                if eval_predictions:
                    image_annotations = get_annotations_by_image_name(filename, dataset)
                    evaluations = evaluate_predictions(
                        image_annotations,
                        predictions,
                        eval_board=True,
                        eval_num_pieces=False,
                        eval_corners=False,
                        eval_bboxes=True,
                        verbose=False,
                    )
                    with lock:
                        image_evaluations.append({"image_path": image_path, **evaluations})
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image_worker, filename) for filename in image_files]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Thread error: {e}")

    while not results_queue.empty():
        output.append(results_queue.get())
    
    output.sort(key=lambda x: x.get("image", ""))

    if eval_predictions:
        # the corners we define are different from the corners in the annotations
        # so there is also a threshold for the error being too small
        TOO_LARGE_CORNER_ERROR = 50000
        TOO_SMALL_CORNER_ERROR = 20000
        evaluations = pd.DataFrame(image_evaluations)
        evaluations["clickable_image_path"] = evaluations["image_path"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        sorted_evals = evaluations  # apply a filter if desired
        sorted_evals = sorted_evals.sort_values(by="board", ascending=False)

        if show_all_images and output_config.get("corners", False):
            for image_path in sorted_evals["clickable_image_path"]:
                cv2.imshow(
                    image_path,
                    cv2.resize(
                        cv2.imread(f"output_images/{image_path}/{image_path}_corners.jpg"),
                        (800, 800),
                    ),
                )
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print(sorted_evals)
        bbox_results = sorted_evals["bboxes"]
        board_results = sorted_evals["board"]
        print(f"Mean of the bounding box score results: '{bbox_results.mean():.4f}', median: {bbox_results.median():.4f}")
        print(f"Mean of the board score results: '{board_results.mean():.4f}', median: {board_results.median():.4f}")

    json.dump(output, open('output.json', 'w'), indent=4)

    print("Output JSON file created.")
    print(f"All images processed. Results saved to {output_dir}")


def process_input(output_dir, output_config, is_delivery: bool = False, eval_predictions: bool = True):
    if is_delivery and eval_predictions:
        raise ValueError("""Do not evaluate predictions in the delivery!
                         In that case, we would be accessing files that do not even exist""")

    if not os.path.exists('input.json'):
        print("input.json file not found.")
        exit(1)

    data = json.load(open("input.json", "r"))
    
    if eval_predictions:
        dataset = get_dataset()

    output = []
    for image in data['image_files']:
        image_path = image  # Json should specify specific path
        try:
            predictions = process_image(image_path, output_dir, output_config, is_delivery=is_delivery)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            if is_delivery:
                continue
            else:
                raise e

        output.append({
            "image": image_path,
            "num_pieces": predictions['num_pieces'],
            "board": predictions['board'],
            "detected_pieces": predictions['detected_pieces'],
        })
        if eval_predictions:
            image_name = os.path.basename(image)
            image_annotations = get_annotations_by_image_name(image_name, dataset)
            evaluations = evaluate_predictions(
                image_annotations,
                predictions,
                eval_board=True,
                eval_num_pieces=False,
                eval_corners=False,
                eval_bboxes=True,
                verbose=True,
            )
            print(evaluations)
    
    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4)

    print("Output JSON file created.")


def stitch_images(output_dir, image_type='warped',  grid_size=None, output_filename=None):
    target_images = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(f'_{image_type}.jpg'):
                target_images.append(os.path.join(root, file))
    
    if not target_images:
        print(f"No {image_type} images found.")
        return None
    
    target_images.sort()
    
    if output_filename is None:
        output_filename = f"stitched_{image_type}_images.jpg"
    
    images = []
    max_height, max_width = 0, 0
    
    for img_path in target_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        images.append(img)
        h, w = img.shape[:2]
        max_height = max(max_height, h)
        max_width = max(max_width, w)
    
    if not images:
        print("No images could be loaded.")
        return None
    
    if grid_size is None:
        total_images = len(images)
        grid_cols = math.ceil(math.sqrt(total_images))
        grid_rows = math.ceil(total_images / grid_cols)
        grid_size = (grid_rows, grid_cols)
    else:
        grid_rows, grid_cols = grid_size
    
    canvas_height = grid_rows * max_height
    canvas_width = grid_cols * max_width
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        if idx >= grid_rows * grid_cols:
            print(f"Warning: Grid size {grid_size} is too small for {len(images)} images. Some images will be omitted.")
            break
        
        row = idx // grid_cols
        col = idx % grid_cols
        
        resized_img = cv2.resize(img, (max_width, max_height))
        
        y_offset = row * max_height
        x_offset = col * max_width
        
        canvas[y_offset:y_offset + max_height, x_offset:x_offset + max_width] = resized_img

    # reduce image by 50%
    scale_factor = 0.5
    canvas = cv2.resize(canvas, (int(canvas.shape[1] * scale_factor), int(canvas.shape[0] * scale_factor)))
    
    output_path = os.path.join(output_dir, output_filename)
    canvas = cv2.resize(canvas, (3000, 3000))   # Resize canvas to 3000x3000 (otherwise it is too large)
    cv2.imwrite(output_path, canvas)
    
    print(f"Stitched {len(images)} {image_type} images into grid {grid_size} at {output_path}")
    return output_path

def main(process_all: bool = True):
    # --- Delete output directory if it exists ---
    output_dir = "output_images"
    if os.path.exists(output_dir):
        print(f"Deleting existing output directory: {output_dir}")
        import shutil

        try:
            shutil.rmtree(output_dir)
            print(f"Successfully deleted output directory: {output_dir}")
        except Exception as e:
            print(f"Error deleting output directory: {e}")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Configure output options ---
    output_config = {
        'original': False,
        'corners': False,
        'threshold': False,
        'warped': False,
        'warped_color': False,
        'clahe': False,
        'blurred_warp': False,
        'canny_edges': False,
        'dilated': False,
        'hough_lines': False,
        'hough_lines_rectified': False,
        'filtered_intersections': False,
        'pieces_gc': True,
        'grabcut_mask': True,
        'grabcut_fg_ratios': True,
        'grabcut_hint_mask': True,
        'otsu_mask': True,
        'black_pieces_mask': True,
        'black_pieces_mask_morph': True,
        'white_pieces_mask': True,
        'white_pieces_mask_morph': True,
        'piece_contours': True,
        'bboxes': True,
        "black_hue_mask": True,
        "black_sat_mask": True,
        "black_val_mask": True,
        "white_hue_mask": True,
        "white_sat_mask": True,
        "white_val_mask": True,
        'bboxes_orig': True,
        'pieces': True,
        'horse': False,
        'rotated': False,
        'bgr_clahed': True,
        'dist_transform_sure_fg': True,
        'sure_bg': True,
        'unknown': True,
        'markers': True,
    }
    for config in output_config:
        for other_config in output_config:
            if config != other_config and other_config.endswith(config):
                raise ValueError(
                    f"""Output config name '{other_config}' ends with the name of '{config}'.
                    No config's name should end with another config's name.""")

    if process_all:
        process_all_images(output_dir, output_config, eval_predictions=True)
    else:
        process_input(output_dir, output_config, is_delivery=False, eval_predictions=True)

    for key, value in output_config.items():
        if value:
            stitch_images(output_dir, image_type=key)

if __name__ == "__main__":
    # process_input(output_dir=None, output_config={}, is_delivery=True, eval_predictions=False)

    main(process_all=True)

    # Test if labels are being correctly loaded (not used for delivery, just for testing)
    # dataset = get_dataset()
    # show_all_annotations(dataset)
