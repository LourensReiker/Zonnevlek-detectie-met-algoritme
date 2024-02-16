#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:33:41 2023

@author: johannesreiker
"""
import cv2
import numpy as np

# Load the input image
input_image = cv2.imread('zon.png')

# Convert the image to grayscale for processing
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

# Apply adaptive thresholding to create a binary image
thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the input image to draw contours on
output_image = input_image.copy()

# Define a minimum area to consider as a sunspot
min_area = 50

# Loop over the contours and draw rectangles around sunspots
for contour in contours:
    if cv2.contourArea(contour) > min_area:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle with thickness 2

# Save the output image
cv2.imwrite('output_image.jpg', output_image)

# Show the output image
cv2.imshow("Output Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()