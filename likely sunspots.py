#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:17:39 2024

@author: johannesreiker
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('zon2.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Apply adaptive thresholding to binarize the image
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw rectangles on
result_img = img.copy()

# Define the color for the rectangles (in RGB format)
red_color = (255, 0, 0)

# Initialize a counter for rectangles
rectangle_count = 0

# Iterate through the contours, draw rectangles, and count the rectangles
for contour in contours:
    # Calculate the area of each contour
    area = cv2.contourArea(contour)
    
    # Filter contours based on area to exclude large areas (the sun)
    if area < 100:  # Adjust this threshold as needed
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangles only for contours with areas smaller than the threshold
        cv2.rectangle(result_img, (x, y), (x + w, y + h), red_color, 3)
        rectangle_count += 1

# Display the result with rectangles
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print the number of rectangles found
print(f"Number of rectangles (likely sunspots): {rectangle_count}")


