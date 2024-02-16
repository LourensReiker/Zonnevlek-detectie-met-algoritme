#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:41:13 2023

@author: johannesreiker
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('zon2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define lower and upper bounds for the color range
lower = np.array([10, 10, 10])
higher = np.array([250, 250, 250])

# Create a binary mask using inRange
mask = cv2.inRange(img, lower, higher)

# Find contours in the binary mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw red lines on
result_img = img.copy()

# Define the color for the red lines (in RGB format)
green_color = (0, 255, 0)

# Draw red lines over the contours
cv2.drawContours(result_img, contours, -1, green_color, 2)


# Display the result using matplotlib
plt.imshow(result_img)
plt.axis('off')
plt.show()

# Optionally, save the result
cv2.imwrite('result_image.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
