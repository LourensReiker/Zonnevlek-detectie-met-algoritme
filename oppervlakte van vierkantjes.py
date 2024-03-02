#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:14:27 2023

@author: johannesreiker
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('zon2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define lower and upper bounds for the color range
lower = np.array([1, 10, 15])
higher = np.array([223, 250, 250])

# Create a binary mask using inRange
mask = cv2.inRange(img, lower, higher)

# Apply Gaussian smoothing to the binary mask to reduce noise
smoothed_mask = cv2.GaussianBlur(mask, (15, 15), 0)

# Find contours in the smoothed binary mask 
contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw green lines on
result_img = img.copy()

# Define the color for the green lines (in RGB format)
blue_color = (0, 0, 255)

# Initialize a counter for rectangles
rectangle_count = 0

# Initialize a variable to store the total size (area) of all rectangles
total_size = 0

# Iterate through the contours, draw green rectangles, and calculate total size
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    # You can adjust the aspect ratio range to filter out non-rectangular shapes
    if 0.8 <= aspect_ratio <= 1.2:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), blue_color, 3)
        rectangle_count += 1
        
        # Calculate the area of the rectangle and add it to the total size
        rectangle_area = w * h
        total_size += rectangle_area

# Display the result with rectangles
plt.imshow(result_img)
plt.axis('off')
plt.show()

# Print the number of rectangles found and the total size
print(f"Number of rectangles: {rectangle_count}")
print(f"Total size of rectangles: {total_size} square pixels")

