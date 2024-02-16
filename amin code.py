import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('zon.png')

# Define the HSV color range for orange
lower_orange = np.array([1, 1, 18])
upper_orange = np.array([25, 255, 255])

# Create a mask to isolate the specified color range (orange)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img_hsv, lower_orange, upper_orange)

# Find coordinates where the mask is white (within the specified color range)
orange_coordinates = np.column_stack(np.where(mask == 255))

# Check if any coordinates were found
if orange_coordinates.shape[0] > 0:
    # Get the BGR colors at the detected coordinates
    orange_colors = [img[y, x] for y, x in orange_coordinates]

    # Convert the colors to HSV for analysis
    orange_colors_hsv = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0] for color in orange_colors]

    # Find the minimum and maximum HSV values
    min_hsv = np.min(orange_colors_hsv, axis=0)
    max_hsv = np.max(orange_colors_hsv, axis=0)

    # Print the minimum and maximum HSV values
    print("Minimum HSV:", min_hsv)
    print("Maximum HSV:", max_hsv)

    # Display the colors in a plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    color_patches = np.zeros((1, len(orange_colors), 3), dtype=np.uint8)
    color_patches[0, :] = orange_colors
    plt.imshow(cv2.cvtColor(color_patches, cv2.COLOR_BGR2RGB))
    plt.title('Detected Colors')
    plt.axis('off')

    plt.show()
else:
    print("No orange colors detected in the image.")
