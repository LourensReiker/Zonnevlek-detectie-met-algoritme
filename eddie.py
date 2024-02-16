#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:06:17 2023

@author: johannesreiker
"""
import cv2
import numpy as np

# load image 
img = cv2.imread('zon.png')

# create mask for red
lower=np.array((20,40,215))
upper=np.array((100,120,255))
mask = cv2.inRange(img, lower, upper)

# mask the image for viewing
result = img.copy()
result[mask!=255] = (0,0,0)

# separate channels
b,g,r = cv2.split(img)

# get average of channels
ave = cv2.add(b,g,r)/3
ave = ave.astype(np.uint8)

# get min and max for ave
ave_min = np.amin(ave[np.where(mask==255)])
ave_max = np.amax(ave[np.where(mask==255)])

# form min and max masks from ave
mask_min = ave.copy()
mask_min[ave==ave_min] = 255
mask_min[ave!=ave_min] = 0

mask_max = ave.copy()
mask_max[ave==ave_max] = 255
mask_max[ave!=ave_max] = 0

# combine min mask with red mask
red_mask_min = cv2.bitwise_and(mask, mask_min)

# combine max mask with red mask
red_mask_max = cv2.bitwise_and(mask, mask_max)

# get coordinates where masks are white and corresponding color - take first one
min_list = np.argwhere(red_mask_min==255)
min_count = len(min_list)
if min_count != 0:
    min_coord = min_list[0]
    y=min_coord[0]
    x=min_coord[1]
    min_color = img[y:y+1, x:x+1][0][0]
    print("min_red:", min_color)

max_list = np.argwhere(red_mask_max==255)
max_count = len(max_list)
if max_count != 0:
    max_coord = max_list[0]
    y=max_coord[0]
    x=max_coord[1]
    max_color = img[y:y+1, x:x+1][0][0]
    print("max_red:", max_color)


# save results
cv2.imwrite("mandril3_nose_mask.jpg", mask)
cv2.imwrite("mandril3_nose.jpg", result)

# show images
cv2.imshow("mask", mask)
cv2.imshow("result", result)
cv2.imshow("ave", ave)
cv2.imshow("red_mask_min", red_mask_min)
cv2.imshow("red_mask_max", red_mask_max)
cv2.waitKey(0)
cv2.destroyAllWindows()