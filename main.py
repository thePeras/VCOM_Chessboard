import numpy as np
import cv2
import os

# Load Image
img = cv2.imread(os.path.join('./data/images', 'G000_IMG087.jpg'), cv2.IMREAD_GRAYSCALE)

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
    area = cv2.contourArea(cnt)
    if area > largest_area:
        largest_area = area
        largest_contour = cnt

# Draw the largest contour
# cv2.drawContours(img, [largest_contour], 0, (0, 255, 0), 2)

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
        
# Draw the points
cv2.circle(img, top_left, 5, (0, 0, 255), -1)
cv2.circle(img, bottom_left, 5, (0, 0, 255), -1)
cv2.circle(img, top_right, 5, (0, 0, 255), -1)
cv2.circle(img, bottom_right, 5, (0, 0, 255), -1)

# Create me the warp matrix based on the points and the comparison points
warp_matrix = cv2.getPerspectiveTransform(np.float32([top_left, bottom_left, top_right, bottom_right]),
                                          np.float32([top_left_comp, bottom_left_comp, top_right_comp, bottom_right_comp]))

# Apply the warp matrix to the image
warped_img = cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]))

# Show images
cv2.imshow('Original Image', img)
cv2.imshow('Warped Image', warped_img)
cv2.imshow('Global Threshold', th_global)

cv2.waitKey(0)
cv2.destroyAllWindows()
