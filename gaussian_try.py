import contours
import cv2
import numpy as np

# Initialize the color model for human skin
histG_hue = np.zeros(256, dtype=np.uint)


# Function to initialize the color model
def color_model_init():
    global histG_hue
    histG_hue = np.zeros(256, dtype=np.uint)


# Function to build the hand color model
def color_model_build(img):
    global histG_hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    hist_hue = cv2.calcHist([hue], [0], None, [256], [0, 256])
    histG_hue += np.squeeze(hist_hue)


# Function to segment the image
def segment_image(img, hue_mean, hue_std, k):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    lower_bound = hue_mean - k * hue_std
    upper_bound = hue_mean + k * hue_std
    mask = cv2.inRange(hue, lower_bound, upper_bound)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


# Function to postprocess the segmented image
def postprocess_image(segmented_img):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(segmented_img, cv2.MORPH_OPEN, kernel)
    return opening


# Function to extract the contour
def extract_contour(segmented_img):
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(segmented_img)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img


# Function to draw the major axis
def draw_major_axis(img, contours):
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(img, ellipse, (0, 0, 255), 2)


# Example usage
if __name__ == "__main__":
    # Load the image
    img = cv2.imread('./images/red_tshirt.jpg')

    # Initialize the color model
    color_model_init()

    # Build the hand color model
    color_model_build(img)

    # Calculate the mean and standard deviation
    hist_mean = np.mean(histG_hue)
    hist_std = np.std(histG_hue)

    # Segment the image
    segmented_img = segment_image(img, hist_mean, hist_std, k=2)

    # Postprocess the segmented image
    processed_img = postprocess_image(segmented_img)

    # Extract the contour
    contour_img = extract_contour(processed_img)

    # Draw the major axis
    draw_major_axis(contour_img, [contours])  # Contours should be the list of contours

    # Show the images
    cv2.imshow('Original Image', img)
    cv2.imshow('Segmented Image', segmented_img)
    cv2.imshow('Processed Image', processed_img)
    cv2.imshow('Contour Image', contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
