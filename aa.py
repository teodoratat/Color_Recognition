import cv2
import numpy as np


def multilevelThresholding(hue_channel, num_bins):
    # Calculate the histogram
    hist, bins = np.histogram(hue_channel, bins=num_bins, range=(0, 180))

    # Calculate the bin edges
    bin_edges = np.linspace(0, 180, num_bins + 1)

    # Threshold the image based on the bin edges
    thresholded_image = np.zeros_like(hue_channel)

    for i in range(num_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        thresholded_image[(hue_channel >= lower_bound) & (hue_channel < upper_bound)] = i * (255 // (num_bins - 1))

    return thresholded_image


# Load your image
image = cv2.imread('./images/green_dress.jpg')

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue_channel = hsv_image[:, :, 0]

# Apply multilevel thresholding with 10 bins
num_bins = 10
thresholded_hue = multilevelThresholding(hue_channel, num_bins)

# Convert the thresholded hue image to BGR for display
thresholded_hue_bgr = cv2.cvtColor(thresholded_hue, cv2.COLOR_GRAY2BGR)

# Display the thresholded image
cv2.imshow('Multilevel Thresholded Hue Image', thresholded_hue_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
