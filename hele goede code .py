#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:38:34 2024

@author: johannesreiker
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# In[10]:


# Load the original color image
original_img = cv2.imread('zon2.png')
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(figsize=(10, 10))
#plt.subplot(1, 3, 1)
plt.imshow(original_img)
# plt.title('Original Image')
# plt.axis('off')


# Define lower and upper bounds for the color range
# lower = np.array([10, 10, 10])
# higher = np.array([250, 250, 250])

# # Create a binary mask using inRange
# mask = cv2.inRange(original_img, lower, higher)

# # Apply Gaussian smoothing to the binary mask to reduce noise
# smoothed_mask = cv2.GaussianBlur(mask, (15, 15), 0)

# Sobel edge detection function
def sobel_edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Sobel filter in the x and y directions
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Convert the results back to uint8 and combine them
    abs_sobel_x = np.uint8(np.absolute(sobel_x))
    abs_sobel_y = np.uint8(np.absolute(sobel_y))
    combined_sobel = cv2.bitwise_or(abs_sobel_x, abs_sobel_y)

    return combined_sobel


def blob_detection(image, params=None):
    # Set up the detector with default parameters or custom parameters
    detector = cv2.SimpleBlobDetector_create(params if params else cv2.SimpleBlobDetector_Params())

    # Detect blobs
    keypoints = detector.detect(image)

    # Calculate area for each blob separately
    blob_areas = [np.pi * key.size**2 / 4 for key in keypoints]

    # Draw detected blobs as red circles
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (255, 0, 0),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Return the keypoints, blob areas, and the image with keypoints drawn
    return keypoints, blob_areas, img_with_keypoints



# Apply Sobel edge detection
sobel_edges = sobel_edge_detection(original_img)

# # Create a binary mask based on your criteria (e.g., thresholding)
binary_mask = cv2.threshold(sobel_edges, 100, 255, cv2.THRESH_BINARY)[1]

# # Use the binary mask to filter out unwanted areas
# masked_sobel_edges = cv2.bitwise_and(sobel_edges, binary_mask)

plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.subplot(1, 3, 2)
plt.imshow(sobel_edges, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap='gray')


# In[56]:


# Blob detection on the masked Sobel edges with adjusted parameters
blob_params = cv2.SimpleBlobDetector_Params()

# Adjust parameters for more precision
blob_params.filterByColor = True
blob_params.blobColor = 255
blob_params.filterByArea = True
blob_params.minArea = 20   # Adjust as needed
blob_params.maxArea = 2000 # Adjust as needed

blob_params.filterByCircularity = False  # You can experiment with circularity
blob_params.minCircularity = 0.5  # Adjust as needed

blob_params.filterByConvexity = False  # You can experiment with convexity
blob_params.minConvexity = 1  # Adjust as needed

blob_params.filterByInertia = False  # You can experiment with inertia
blob_params.minInertiaRatio = 0.3  # Adjust as needed

kp_sobel, blob_area_sobel, im_sobel_kp = blob_detection(sobel_edges, blob_params)
kp_mask,  blob_area_mask,  im_mask_kp  = blob_detection(binary_mask, blob_params)


# In[57]:


plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1)
plt.imshow(original_img)
plt.subplot(2, 3, 2)
plt.imshow(sobel_edges, cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(binary_mask, cmap='gray')
plt.subplot(2, 3, 5)
plt.imshow(im_sobel_kp, cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(im_mask_kp, cmap='gray')


# In[ ]:


# Print the number of keypoints found and total blob area

    # # Print details of each blob
    # for i, keypoint in enumerate(keypoints):
    #     print(f"Blob {i + 1}:")
    #     print(f"  Position: ({int(keypoint.pt[0])}, {int(keypoint.pt[1])})")
    #     print(f"  Size: {keypoint.size}")
    #     print(f"  Response: {keypoint.response}")
    #     print("---")


print(f"Number of blobs detected on Sobel edges: {len(kp_sobel)}")
for i, area in enumerate(blob_area_sobel):
    print(f"Blob {i + 1} area: {area} square pixels")

print(f"Number of blobs detected on binary mask: {len(kp_mask)}")
for i, area in enumerate(blob_area_mask):
    print(f"Blob {i + 1} area: {area} square pixels")



gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
gray_binary = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)[1]

plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(gray_binary, cmap='gray')


# In[ ]:




