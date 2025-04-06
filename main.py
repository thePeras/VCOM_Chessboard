import numpy as np
import cv2
import os
import json

def process_image(image_path, output_dir):
    # Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Apply a Gaussian blur - To eliminate the noise in segmentation
    blur_img = cv2.GaussianBlur(img, (11, 11), 0)
    
    # Apply global binary threshold - Board segmentation
    ret, th_global = cv2.threshold(blur_img, 200, 255, cv2.THRESH_BINARY)
    
    # Get the contours of the board
    contours, hierarchy = cv2.findContours(th_global, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored drawing
    cv2.drawContours(contour_img, [largest_contour], 0, (0, 0, 255), 3)  # Draw the largest contour in red
    
    # --- Improved Corner Detection ---
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
        print(f"Could not approximate to four points for {image_path} after {max_attempts} attempts")
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
        np.float32([top_left_comp, bottom_left_comp, top_right_comp, bottom_right_comp])
    )

    # Apply the warp matrix to the image
    warped_img = cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]))
    
    # Save images
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image_folder = os.path.join(output_dir, base_filename)
    os.makedirs(image_folder, exist_ok=True)

    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_original.jpg'), img)
    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_corners.jpg'), points_img)
    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_contour.jpg'), contour_img)
    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_warped.jpg'), warped_img)
    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_threshold.jpg'), th_global)

    print(f"Processed {base_filename}")

# Create output directory if it doesn’t exist
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Process all images in the images directory
images_dir = './data/images'

def process_all_images():
    for filename in os.listdir(images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, filename)
            process_image(image_path, output_dir)
    
    print(f"All images processed. Results saved to {output_dir}")


# Uncomment the following line to process a single image
#image_path = PATH_TO_IMAGE
#process_image(image_path, output_dir)


if __name__ == "__main__":
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
            "num_pieces": 0,
            "board": [],
            "detected_pieces": []
        })
    
    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    print("Output JSON file created.")
