import numpy as np
import cv2
import os
from typing import Optional
import json
import pandas as pd
from shapely.geometry import Polygon

from dataset_annotations import (
    evaluate_predictions,
    get_dataset,
    get_annotations_by_image_name,
)

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


def process_image(
    image_path, output_dir: Optional[str] = None, output_config: Optional[dict] = None
):
    THRESHOLD_MAXVAL: int = 255
    THRESHOLD_THRESH: int = 200
    THRESHOLD_SATURATION_MAX: int = 50
    CANNY_LOWER: int = 100
    CANNY_UPPER: int = 250
    MAX_DISTANCE_TO_MERGE_CONTOURS: int = 5
    MIN_DISTANCE_TO_IMAGE_BORDER: int = 10
    FILTER_BY_SATURATION: bool = False

    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)
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
    ret, th_global = cv2.threshold(
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
    warped_img = cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]))

    if output_dir is not None:
        # Save images
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        image_folder = os.path.join(output_dir, base_filename)
        os.makedirs(image_folder, exist_ok=True)
        output_handlers = [
            ("original", lambda: img),
            ("corners", lambda: points_img),
            # ('contour', lambda: contour_img),
            # ('basic_contour', lambda: basic_contour_img),
            ("threshold", lambda: th_global),
            # ("threshold_inverted", lambda: th_inverted),
            ("warped", lambda: warped_img),
            ("canny_edges", lambda: cv2.Canny(warped_img, 100, 200)),
            # ('initial_canny_edges', lambda: canny),
            # ('contours_no_ch', lambda: contour_no_ch_img),
            *base_img_contour_images,
            *threshold_contour_images,
        ]

        for output_type, get_image in output_handlers:
            if output_config.get(output_type, True):
                image = get_image()
                cv2.imwrite(
                    os.path.join(image_folder, f"{base_filename}_{output_type}.jpg"),
                    image,
                )

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


def process_all_images(output_dir, output_config, eval_predictions: bool = True, show_intermediate_images: bool = True):
    images_dir = "./data/images"
    output = []

    if eval_predictions:
        dataset = get_dataset()
        image_evaluations = []

    for filename in os.listdir(images_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(images_dir, filename)
            output.append(
                {
                    "image": image_path,
                    "num_pieces": get_number_of_pieces(image_path),
                    "board": get_board(image_path),
                    "detected_pieces": get_detected_pieces(image_path),
                }
            )

            image_output_dict = process_image(image_path, output_dir, output_config)
            if eval_predictions:
                image_annotations = get_annotations_by_image_name(filename, dataset)
                evaluations = evaluate_predictions(
                    image_annotations,
                    image_output_dict,
                    eval_board=False,
                    eval_num_pieces=False,
                    verbose=True,
                )
                image_evaluations.append({"image_path": image_path, **evaluations})

    if eval_predictions:
        TOO_LARGE_CORNER_ERROR = 50000
        TOO_SMALL_CORNER_ERROR = 20000  # the corners we define are different from the corners in the annotations
        evaluations = pd.DataFrame(image_evaluations)
        evaluations["clickable_image_path"] = evaluations["image_path"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        bad_corners = evaluations[
            (evaluations["corners"] >= TOO_LARGE_CORNER_ERROR) | (evaluations["corners"] <= TOO_SMALL_CORNER_ERROR)
        ]
        bad_corners = evaluations
        bad_corners = bad_corners.sort_values(by="corners", ascending=False)
        if show_intermediate_images:
            for image_path in bad_corners["clickable_image_path"]:
                cv2.imshow(
                    image_path,
                    cv2.resize(
                        cv2.imread(f"output_images/{image_path}/{image_path}_corners.jpg"),
                        (800, 800),
                    ),
                )
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print(f"Possible bad corners count: {len(bad_corners)} (not all are actually wrong)")
        print(bad_corners)

    with open("output.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Output JSON file created.")
    print(f"All images processed. Results saved to {output_dir}")


def process_input(output_dir, output_config):
    if not os.path.exists("input.json"):
        print("input.json file not found.")
        exit(1)

    with open("input.json", "r") as f:
        data = json.load(f)

    output = []
    for image in data["image_files"]:
        image_path = os.path.join("data/", image)  # TODO: Delete data/ on submission
        output.append(
            {
                "image": image_path,
                "num_pieces": get_number_of_pieces(image_path),
                "board": get_board(image_path),
                "detected_pieces": get_detected_pieces(image_path),
            }
        )
        process_image(image_path, output_dir, output_config)

    with open("output.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Output JSON file created.")


if __name__ == "__main__":
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
        "original": True,
        "corners": True,
        "contour": True,
        "threshold": True,
        "warped": True,
        "canny_edges": True,
        "merged_lines": True,
    }

    process_all_images(output_dir, output_config)
    # process_input(output_dir, output_config)
