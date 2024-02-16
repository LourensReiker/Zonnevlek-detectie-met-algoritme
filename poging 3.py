import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('zon3.png.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define lower and upper bounds for the color range
lower = np.array([10, 10, 10])
higher = np.array([250, 250, 250])

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

# Draw green lines over the smoothed contours
cv2.drawContours(result_img, contours, -1, blue_color, 2)

# Iterate through the contours and draw green rectangles
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result_img, (x, y), (x + w, y + h), blue_color, 3)

# Display the result using matplotlib
plt.imshow(result_img)
plt.axis('off')
plt.show()

# Optionally, save the result
cv2.imwrite('result_image.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

