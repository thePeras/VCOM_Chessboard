import numpy as np
import cv2
import os

def process_image(image_path, output_dir):
    # Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply a Gaussian blur - To eliminate the noise in segmentation
    blur_img = cv2.GaussianBlur(img, (11, 11), 0)
    
    # Apply global binary threshold - Board segmentation
    ret, th_global = cv2.threshold(blur_img, 200, 255, cv2.THRESH_BINARY)
    
    # Get the contours of the board
    contours, hierarchy = cv2.findContours(th_global, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        print(f"No contours found in {image_path}")
        return
    
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored drawing
    cv2.drawContours(contour_img, [largest_contour], 0, (0, 0, 255), 3)  # Draw the largest contour in red
    

    # Get the bounding box of the largest contour
    #x, y, w, h = cv2.boundingRect(largest_contour)
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Find the closest point of the countour to the (0, 0) point
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    top_left  = largest_contour[0][0]
    top_left_comp = (0, 0)
    bottom_left = largest_contour[0][-1]
    bottom_left_comp = (0, img.shape[0])
    top_right = largest_contour[0][0]
    top_right_comp = (img.shape[0], 0)
    bottom_right = largest_contour[0][-1]
    bottom_right_comp = (img.shape[0], img.shape[0])
    
    for i in range(len(largest_contour)):
        point = largest_contour[i][0]
        if euclidean_distance(point, top_left_comp) < euclidean_distance(top_left, top_left_comp):
            top_left = point
        if euclidean_distance(point, bottom_left_comp) < euclidean_distance(bottom_left, bottom_left_comp):
            bottom_left = point
        if euclidean_distance(point, top_right_comp) < euclidean_distance(top_right, top_right_comp):
            top_right = point
        if euclidean_distance(point, bottom_right_comp) < euclidean_distance(bottom_right, bottom_right_comp):
            bottom_right = point
    
    # Draw the points on a copy of the original image
    points_img = img.copy()
    points_img = cv2.cvtColor(points_img, cv2.COLOR_GRAY2BGR)
    cv2.circle(points_img, tuple(top_left), 5, (0, 0, 255), -1)
    cv2.circle(points_img, tuple(bottom_left), 5, (0, 0, 255), -1)
    cv2.circle(points_img, tuple(top_right), 5, (0, 0, 255), -1)
    cv2.circle(points_img, tuple(bottom_right), 5, (0, 0, 255), -1)
    
    # Create the warp matrix based on the points and the destination points
    warp_matrix = cv2.getPerspectiveTransform(np.float32([top_left, bottom_left, top_right, bottom_right]),
                                            np.float32([top_left_comp, bottom_left_comp, top_right_comp, bottom_right_comp]))

    # Apply the warp matrix to the image
    warped_img = cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]))
    
    # Save images
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image_folder = os.path.join(output_dir, base_filename)
    os.makedirs(image_folder, exist_ok=True)

    cv2.imwrite(os.path.join(image_folder,  f'{base_filename}_original.jpg'), img)
    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_corners.jpg'), points_img)
    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_contour.jpg'), contour_img)
    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_warped.jpg'), warped_img)
    cv2.imwrite(os.path.join(image_folder, f'{base_filename}_threshold.jpg'), th_global)

    print(f"Processed {base_filename}")


# Create output directory if it doesn't exist
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Process all images in the images directory
images_dir = './data/images'
for filename in os.listdir(images_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_dir, filename)
        process_image(image_path, output_dir)

print(f"All images processed. Results saved to {output_dir}")