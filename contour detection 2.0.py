#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:54:07 2023

@author: johannesreiker
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('zon2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define lower and upper bounds for the color range
lower = np.array([1, 1, 1])
higher = np.array([100, 100, 100])

# Create a binary mask using inRange
mask = cv2.inRange(img, lower, higher)

plt.imshow(mask, 'gray')