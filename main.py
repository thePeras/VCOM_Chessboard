import numpy as np
import cv2
import os
from typing import Optional
import json

from dataset_annotations import (
    evaluate_predictions,
    get_dataset,
    get_annotations_by_image_name,
)

def process_image(image_path, output_dir: Optional[str] = None, output_config: Optional[dict] = None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Apply a Gaussian blur - To eliminate the noise in segmentation
    blur_img = cv2.GaussianBlur(img, (11, 11), 0)

    # Apply global binary threshold - Board segmentation
    ret, th_global = cv2.threshold(blur_img, 200, 255, cv2.THRESH_BINARY)

    # Get the contours of the board
    contours, hierarchy = cv2.findContours(
        th_global, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print(f"No contours found in {image_path}")
        return

    # Get the largest contour
    largest_area = 0
    largest_contour = None
    for cnt in contours:
        contour_ch = cv2.convexHull(cnt)
        area = cv2.contourArea(contour_ch)
        if area > largest_area:
            largest_area = area
            largest_contour = contour_ch

    if largest_contour is None:
        print(f"No valid contour found in {image_path}")
        return
    
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  
    cv2.drawContours(contour_img, [largest_contour], 0, (0, 0, 255), 3)  # Draw the largest contour in red
    
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
        cv2.circle(points_img, tuple(map(int, pt)), 15, (0, 0, 255), -1)

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
    warped_img = cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]))

    if output_dir is not None:
        # Save images
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        image_folder = os.path.join(output_dir, base_filename)
        os.makedirs(image_folder, exist_ok=True)
        output_handlers = [
            ('original', lambda: img),
            ('corners', lambda: points_img),
            ('contour', lambda: contour_img),
            ('threshold', lambda: th_global),
            ('warped', lambda: warped_img),
            ('canny_edges', lambda: cv2.Canny(warped_img, 100, 200))
        ]
        for output_type, get_image in output_handlers:
            if output_config.get(output_type, False):
                image = get_image()
                cv2.imwrite(os.path.join(image_folder, f'{base_filename}_{output_type}.jpg'), image)

    board = [[0] * 8 for _ in range(8)]  # placeholder for board
    predictions = {
        "image": image_path,
        "corners": {
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
            "top_left": top_left,
            "top_right": top_right,
        },
        "board": board,
        "detected_pieces": [],
        "num_pieces": sum([sum(row) for row in board]),
    }

    print()
    print(f"Processed {base_filename}")
    return predictions

# Position of the pieces on the board (8x8 matrix with 0/1 values)
def get_board(image_path):
    # TODO
    return []

# Position of the pieces on the image (bounding boxes)
def get_detected_pieces(image_path):
    # TODO
    return []

# Total number of black/white pieces on the board
def get_number_of_pieces(image_path):
    # TODO
    # This method is just an intersection of the board with the detected pieces bounding boxes
    return 0

def process_all_images(output_dir, output_config, evaluate_predictions: bool = True):
    images_dir = './data/images'
    output = []
    for filename in os.listdir(images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, filename)
            output.append({
                "image": image_path,
                "num_pieces": get_number_of_pieces(image_path),
                "board": get_board(image_path),
                "detected_pieces": get_detected_pieces(image_path),
            })

            image_output_dict = process_image(image_path, output_dir, output_config)
            if evaluate_predictions:
                image_annotations = get_annotations_by_image_name(filename, dataset)
                evaluations = evaluate_predictions(
                    image_annotations,
                    image_output_dict,
                    eval_board=False,
                    eval_num_pieces=False,
                    verbose=True,
                )
                print(evaluations)

    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print("Output JSON file created.")
    print(f"All images processed. Results saved to {output_dir}")

def process_input(output_dir, output_config):
    if not os.path.exists('input.json'):
        print("input.json file not found.")
        exit(1)

    with open('input.json', 'r') as f:
        data = json.load(f)

    output = []
    for image in data['image_files']:
        image_path = os.path.join("data/", image) # TODO: Delete data/ on submission
        output.append({
            "image": image_path,
            "num_pieces": get_number_of_pieces(image_path),
            "board": get_board(image_path),
            "detected_pieces": get_detected_pieces(image_path),
        })
        process_image(image_path, output_dir, output_config)
    
    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print("Output JSON file created.")


if __name__ == "__main__":
    # --- Delete output directory if it exists ---
    output_dir = 'output_images'
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
        'original': True,
        'corners': False,
        'contour': False,
        'threshold': False,
        'warped': True,
        'canny_edges': True,
        'merged_lines': True
    }

    process_all_images(output_dir, output_config)
    #process_input(output_dir, output_config)

