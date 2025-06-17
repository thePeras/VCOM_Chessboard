import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def draw_keypoints_on_image(image, keypoints, color=(255, 0, 255)):
    for person_keypoints in keypoints:
        points = person_keypoints.reshape(-1, 3)  # (num_kpts, (x, y, confidence))
        for x, y, conf in points:
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 20, color, -1)

    return image

def draw_bounding_boxes(image, boxes, color=(255, 0, 0), thickness=10):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def main(image_path, model_path='best.pt'):
    # Load model
    model = YOLO(model_path)

    # Run inference
    results = model(image_path)

    # Get original image
    image = cv2.imread(image_path)

    # Extract keypoints from results
    keypoints = results[0].keypoints.xy.cpu().numpy()  # shape: (n_people, num_kpts, 2)
    keypoint_conf = results[0].keypoints.conf.cpu().numpy()  # shape: (n_people, num_kpts)
    keypoints_with_conf = [
        np.concatenate([xy, c[:, None]], axis=1)  # combine xy and conf
        for xy, c in zip(keypoints, keypoint_conf)
    ]
    
    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: (n_people, 4) [x1, y1, x2, y2]

    # Draw keypoints and bounding boxes
    output_image = draw_keypoints_on_image(image.copy(), keypoints_with_conf)
    output_image = draw_bounding_boxes(output_image, boxes)

    # Show image
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig("output_yolopose.png")


if __name__ == "__main__":
    main("dataset/images/58/G058_IMG037.jpg", "models/pose/best_3.pt")  #

