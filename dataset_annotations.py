import json
import os
import cv2
import numpy as np
from typing import Optional


# print(dataset.keys())
# print()

# print(dataset["images"][0])

# print()
# print(dataset["annotations"].keys())

# print()
# print(dataset["annotations"]["corners"][0])
# print(dataset["annotations"]["corners"][0].keys())

# print()
# print(dataset["annotations"]["pieces"][0])
# print(dataset["annotations"]["pieces"][0].keys())

# print()
# print(len(dataset["images"]))
# images = dataset["images"]

# distinct_names = set(images["file_name"] for images in dataset["images"])
# print(len(distinct_names))
# assert(len(distinct_names) == len(dataset["images"]))
# print()


def get_piece_position(piece: str):
    """
    Get the position of the piece.
    """
    piece = piece.lower()
    row = ord(piece[0]) - ord("a")
    col = ord(piece[1]) - ord("1")
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
    return None

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
        pass

    return image


def draw_corners(image, image_annotations, predictions: Optional[list] = None):
    """
    Show corners of the image.
    """
    for corner in image_annotations["corners"].values():
        cv2.circle(image, (int(corner[0]), int(corner[1])), 20, (0, 255, 0), -1)

    if predictions is not None:
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


def evaluate_predictions(
    image_annotations,
    predictions,
    eval_corners: bool = True,
    eval_num_pieces: bool = True,
    eval_board: bool = True,
    verbose: bool = True,
):
    true_board = image_annotations["board"]
    true_num_pieces = sum([sum(row) for row in true_board])
    # true_bboxs = image_annotations["detected_pieces"] # don't need this for now

    pred_board = predictions["board"]
    pred_num_pieces = sum([sum(row) for row in pred_board])
    # pred_bboxs = predictions["detected_pieces"]   # don't need this for now

    num_pieces_diff = 0
    board_diff = 0
    corners_mse = 0
    if eval_num_pieces:
        # Eval number of pieces
        num_pieces_diff = abs(true_num_pieces - pred_num_pieces)
        if verbose:
            print(f"Num pieces diff: {num_pieces_diff}")

    if eval_board:
        # Now eval the board
        board_diff = 0
        for i in range(8):
            for j in range(8):
                if true_board[i][j] != pred_board[i][j]:
                    board_diff += 1
        if verbose:
            print(f"Board diff: {board_diff}")

    if eval_corners:
        # Now eval the corners
        corners_mse = evaluate_corners(image_annotations["corners"], predictions["corners"])
        # for corner_name in corner_names:
        #     true_corner = [corner_name]
            
        #     pred_corner = predictions["corners"][corner_name]
        #     corners_mse += (true_corner[0] - pred_corner[0]) ** 2 + (
        #         true_corner[1] - pred_corner[1]
        #     ) ** 2
        # corners_mse = corners_mse / len(corner_names)
        # corners_mse = corners_mse**0.5
        if verbose:
            print(f"Corners MSE: {corners_mse:.2f}")

    return {
        "num_pieces": num_pieces_diff,
        "board": board_diff,
        "corners": corners_mse,
    }


def run_all(dataset):
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


if __name__ == "__main__":
    dataset = get_dataset()
    run_all(dataset)
