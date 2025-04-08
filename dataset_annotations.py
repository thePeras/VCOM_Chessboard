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


def get_image_id_by_name(image_name):
    """
    Get image id by image name.
    """
    for image in dataset["images"]:
        if image["file_name"] == image_name:
            return image["id"]
    return None


def get_annotations_by_image_name(image_name, corner_annotations, piece_annotations):
    """
    Get annotations by image name.
    """
    image_id = get_image_id_by_name(image_name)
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


def draw_annotations(image_name, image_path, corner_annotations, piece_annotations):
    image_annotations = get_annotations_by_image_name(
        image_name, corner_annotations, piece_annotations
    )

    image = cv2.imread(image_path)
    image = draw_bboxes(image, image_annotations, [])
    image = draw_corners(image, image_annotations, [])
    image = cv2.resize(image, (800, 800))

    return image


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
        corner_names = ["bottom_left", "bottom_right", "top_left", "top_right"]

        corners_mse = 0
        for corner_name in corner_names:
            true_corner = image_annotations["corners"][corner_name]
            pred_corner = predictions["corners"][corner_name]
            corners_mse += (true_corner[0] - pred_corner[0]) ** 2 + (
                true_corner[1] - pred_corner[1]
            ) ** 2
        corners_mse = corners_mse / len(corner_names)
        corners_mse = corners_mse**0.5
        if verbose:
            print(f"Corners MSE: {corners_mse}")

    return num_pieces_diff, board_diff, corners_mse


def run_all(corner_annotations, piece_annotations):
    image_list = list(sorted(os.listdir(os.path.join("data", "images"))))

    image_paths = {
        image_name: os.path.join("data", "images", image_name)
        for image_name in image_list
        if image_name.endswith(".jpg")
    }
    # don't need this for now
    # image_annotations = {
    #     image_name: get_annotations_by_image_name(image_name, corner_annotations, piece_annotations)
    #     for image_name in image_list
    # }

    output_dir = "annotated_images"
    os.makedirs(output_dir, exist_ok=True)
    for image_name in image_list:
        image_path = image_paths[image_name]
        # image_annotation_info = image_annotations[image_name]

        output_path = os.path.join(output_dir, image_name)
        result_image = draw_annotations(
            image_name, image_path, corner_annotations, piece_annotations
        )
        cv2.imwrite(output_path, result_image)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    with open("complete_dataset/annotations.json", "r") as f:
        dataset = json.load(f)

    corner_annotations = dataset["annotations"]["corners"]
    piece_annotations = dataset["annotations"]["pieces"]
    run_all(corner_annotations, piece_annotations)
