import math
import cv2
import numpy as np
from shapely.geometry import Polygon
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

##==================================== Square and Piece Detection Helpers ====================================================##

def filter_and_rectify_hough_lines(lines, angle_threshold=10, distance_threshold=20):

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

def identify_and_add_missing_lines(verticals, horizontals, max_gap_ratio=1.8):
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
    ensuring each point is at least 270 pixels away from any other selected point.
    """
    MIN_DISTANCE_BETWEEN_2_POINTS = 270

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
            if distance < MIN_DISTANCE_BETWEEN_2_POINTS:
                valid_point = False
                break
        
        if valid_point:
            filtered_intersections.append(point)
            
        if len(filtered_intersections) == 81:
            break
    
    square_side = int(math.sqrt(len(filtered_intersections)))

    if len(filtered_intersections) < 81:
        print(
            f"Warning: Only found {len(filtered_intersections)} valid intersections with minimum distance of {MIN_DISTANCE_BETWEEN_2_POINTS} pixels"
        )
    
    return filtered_intersections, square_side

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
    contours, _ = cv2.findContours(
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

##==================================== Horse to detect the chessboard orientation ================================================##
def find_orientation(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image.copy()

    # Safely unpack height and width
    height, width = img_gray.shape[:2]

    horse_path = "figures/horse.png"
    horse_img = cv2.imread(horse_path, cv2.IMREAD_GRAYSCALE)
    
    if horse_img is None:
        print("Could not load the horse template image from:", horse_path)
        return None, None

    corner_size = min(width, height) // 4
    orientations = {}
    orientations["bottom_left"] = (
        img_gray[height - corner_size : height, 0:corner_size],
        horse_img,
    )

    orientations["top_left"] = (
        img_gray[0:corner_size, 0:corner_size],
        cv2.rotate(horse_img, cv2.ROTATE_90_CLOCKWISE),
    )

    orientations["top_right"] = (
        img_gray[0:corner_size, width - corner_size : width],
        cv2.rotate(horse_img, cv2.ROTATE_180),
    )

    orientations["bottom_right"] = (
        img_gray[height - corner_size : height, width - corner_size : width],
        cv2.rotate(horse_img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )

    best_score = -1.0
    best_match_loc = None
    best_rotation = None

    for corner_name, (corner_img, horse_template) in orientations.items():
        target_size = corner_size // 4
        if target_size < 1:
            continue

        h_t, w_t = horse_template.shape[:2]

        scale = target_size / max(h_t, w_t)
        new_w = max(1, int(w_t * scale))
        new_h = max(1, int(h_t * scale))
        resized_template = cv2.resize(horse_template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if corner_img.shape[0] < resized_template.shape[0] or corner_img.shape[1] < resized_template.shape[1]:
            continue

        result = cv2.matchTemplate(corner_img, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            if corner_name == "bottom_left":
                match_x = max_loc[0]
                match_y = height - corner_size + max_loc[1]
                best_rotation = None 
            elif corner_name == "top_left":
                match_x = max_loc[0]
                match_y = max_loc[1]
                best_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif corner_name == "top_right":
                match_x = width - corner_size + max_loc[0]
                match_y = max_loc[1]
                best_rotation = cv2.ROTATE_180
            elif corner_name == "bottom_right":
                match_x = width - corner_size + max_loc[0]
                match_y = height - corner_size + max_loc[1]
                best_rotation = cv2.ROTATE_90_CLOCKWISE

            best_match_loc = (match_x, match_y)

    return best_rotation, best_match_loc


def get_board_cornes(
    image_path,
):
    # For corner detection
    THRESHOLD_MAXVAL: int = 255
    THRESHOLD_THRESH: int = 200
    CANNY_LOWER: int = 100
    CANNY_UPPER: int = 250
    MAX_DISTANCE_TO_MERGE_CONTOURS: int = 5
    MIN_DISTANCE_TO_IMAGE_BORDER: int = 10

    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"Failed to load image: {image_path}")
        return


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

    # Extract the four points
    return [pt[0] for pt in approx]

def get_board_squares(corners, img, img_color, image_path):
    # Order points: top-left, top-right, bottom-left, bottom-right
    points = corners
    points.sort(key=lambda p: p[1])  # Sort by y-coordinate
    top_points = points[:2]
    bottom_points = points[2:4]
    top_points.sort(key=lambda p: p[0])  # Sort top points by x-coordinate
    bottom_points.sort(key=lambda p: p[0])  # Sort bottom points by x-coordinate
    top_left, top_right = top_points
    bottom_left, bottom_right = bottom_points

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

    filtered_intersections = []
    if lines is not None:
        verticals, horizontals = filter_and_rectify_hough_lines(
            lines, angle_threshold=10, distance_threshold=20
        )
        
        verticals, horizontals = identify_and_add_missing_lines(verticals, horizontals)
        
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

        if intersections != []:
            x_center, y_center = warped_gray_img.shape[1] // 2, warped_gray_img.shape[0] // 2

            filtered_intersections, square_side = filter_intersections_by_distance(intersections, (x_center, y_center))
            for point in filtered_intersections:
                cv2.circle(filtered_intersections_img, point, 25, (0, 255, 0), -1)

    else:
        print(f"No lines detected in {image_path}")

    filtered_intersections.sort(key=lambda p: (p[1], p[0]))

    found_all_intersections = None
    if len(filtered_intersections) >= 81:
        rows = cols = 8
        step = 9
        found_all_intersections = True
    else:
        grid_side = int(math.sqrt(len(filtered_intersections))) - 1
        rows = cols = grid_side if grid_side > 0 else 0
        step = grid_side + 1
        print(f"Using a {rows}x{cols} grid with {len(filtered_intersections)} intersections")
        found_all_intersections = False

    return filtered_intersections, found_all_intersections, warp_matrix, warped_color_img

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


def get_ground_truth_fen(filename):
    with open('annotations_fen.json', 'r') as f:
        data = json.load(f)
    fen_str = data[str(filename)]
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
    """
    Rotates a 2D matrix based on OpenCV rotation flags.

    Args:
        rotation (int): The rotation flag from cv2 (e.g., cv2.ROTATE_90_CLOCKWISE).
        board_matrix (list of lists): The 8x8 matrix representing the chessboard.

    Returns:
        list of lists: The rotated matrix.
    """
    # Convert the board matrix to a NumPy array for efficient rotation
    np_matrix = np.array(board_matrix)

    # Determine the number of 90-degree rotations (k) for np.rot90
    # k=1 rotates 90 degrees counter-clockwise
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        # One 90-degree clockwise rotation is equivalent to three counter-clockwise rotations
        rotated_matrix = np.rot90(np_matrix, k=-1)
    elif rotation == cv2.ROTATE_180:
        # A 180-degree rotation
        rotated_matrix = np.rot90(np_matrix, k=2)
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        # A 90-degree counter-clockwise rotation
        rotated_matrix = np.rot90(np_matrix, k=1)
    else:
        # If no rotation is needed, return the original matrix
        return board_matrix

    # Convert the rotated NumPy array back to a list of lists
    return rotated_matrix.tolist()



# ==================================== Main Execution ====================================================

#model_path = "models/myYolov8n/weights/best.pt"
model_path = "models/myYolo11s/weights/best.pt"
image_directory = "data/images/"
model = YOLO(model_path)


def process_all_images():
    edit_distances = []
    csv_filepath = "mismatches/task3/mismatch_log.csv"
    csv_headers = ['filename', 'predicted_fen', 'true_fen', 'edit_distance']

    # --- Main Processing Loop ---
    # Use 'with open' to handle the CSV file safely. It's opened once for efficiency.
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        # Create a writer object and write the header row
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)

        # Iterate over all images in the directory
        for filename in os.listdir(image_directory):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue # Skip non-image files

            try:
                image_path = os.path.join(image_directory, filename)
                print(f"\n--- Processing {filename} ---")

                # 1. Predict pieces
                print("Predicting pieces...")
                results = model.predict(image_path, verbose=False)
                yolo_result = results[0]

                # 2. Detect corners
                print("Detecting chessboard corners...")
                corners = get_board_cornes(image_path)
                if not corners or len(corners) != 4:
                    print("Error: Could not find chessboard corners. Skipping file.")
                    continue

                # 3. Rectify perspective
                print("Correcting perspective and finding grid...")
                img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                intersections, all_found, warp_matrix, warped_img = get_board_squares(corners, img_gray, img_color, image_path)

                if not intersections:
                    print("Error: Could not determine the board grid. Skipping file.")
                    continue

                # 4. Map pieces to board
                print("Mapping pieces to squares...")
                board_matrix = map_pieces_to_board(yolo_result, intersections, warp_matrix, FEN_MAP)

                # 5. Determine orientation
                print("Determining board orientation...")
                rotation, _ = find_orientation(warped_img)
                if rotation in [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    board_matrix = rotate_matrix(rotation, board_matrix)

                # 6. Compute FEN and get edit distance
                # Assumes process_fen returns a dict with 'edit_dist', 'fen', and 'true_fen'
                result = process_fen(board_matrix, filename)
                current_edit_dist = result.get('edit_dist', -1)
                edit_distances.append(current_edit_dist)

                # Check if there was a prediction error
                if current_edit_dist != 0:
                    print(f"Prediction error found! Edit distance: {current_edit_dist}. Logging mismatch.")

                    # --- Log to CSV File ---
                    true_fen = result.get('exp_fen', 'N/A')
                    predicted_fen = result.get('pred_fen', 'N/A')
                    row_data = [filename, predicted_fen, true_fen, current_edit_dist]
                    csv_writer.writerow(row_data)

            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")
                # Optionally log errors to the CSV as well
                csv_writer.writerow([filename, 'ERROR', str(e), -1])


    # --- Final Report ---
    if edit_distances:
        # Filter out any potential -1 error values before calculating mean
        valid_distances = [d for d in edit_distances if d != -1]
        if valid_distances:
            mean_dist = np.mean(valid_distances)
            print(f'\nMean edit distance across {len(valid_distances)} images: {mean_dist:.4f}')
        else:
            print("\nNo valid edit distances were calculated.")
    else:
        print("\nNo images were processed.")

    print(f"Mismatch log saved to: {csv_filepath}")


def process_single_image(filename):
    image_path = os.path.join(image_directory, filename)
    print(f"\n--- Processing {filename} ---")

    # 1. Predict pieces
    print("Predicting pieces...")
    results = model.predict(image_path, verbose=False)
    yolo_result = results[0]

    # 2. Detect corners
    print("Detecting chessboard corners...")
    corners = get_board_cornes(image_path)
    if not corners or len(corners) != 4:
        print("Error: Could not find chessboard corners. Skipping file.")

    # 3. Rectify perspective
    print("Correcting perspective and finding grid...")
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    intersections, all_found, warp_matrix, warped_img = get_board_squares(corners, img_gray, img_color, image_path)

    if not intersections:
        print("Error: Could not determine the board grid. Skipping file.")

    # 4. Map pieces to board
    print("Mapping pieces to squares...")
    board_matrix = map_pieces_to_board(yolo_result, intersections, warp_matrix, FEN_MAP)

    # 5. Determine orientation
    print("Determining board orientation...")
    rotation, _ = find_orientation(warped_img)
    if rotation in [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        board_matrix = rotate_matrix(rotation, board_matrix)

    # 6. Compute FEN and get edit distance
    # Assumes process_fen returns a dict with 'edit_dist', 'fen', and 'true_fen'
    result = process_fen(board_matrix, filename)
    print(result)





process_all_images()
#process_single_image("G083_IMG073.jpg")

"""
board = chess.Board(fen_string)
fen_svg = chess.svg.board(board=board)
output_filename = "chessboard_render.png"
cairosvg.svg2png(bytestring=fen_svg.encode('utf-8'), write_to=output_filename)
print(f"Chessboard image saved to {output_filename}")
"""