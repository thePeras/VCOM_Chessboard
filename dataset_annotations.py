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
