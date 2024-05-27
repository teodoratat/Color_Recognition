import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt


def resizeImage(img, max_width, max_height):
    height, width = img.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
    return img


def remove_background(img):
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    contour_mask = np.zeros_like(markers, dtype=np.uint8)
    contour_mask[markers > 1] = 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    ROI = cv2.bitwise_and(original, original, mask=mask)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_ROI = ROI[y:y + h, x:x + w]
    cropped_mask = mask[y:y + h, x:x + w]
    result = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(3):
        result[:, :, i] = cropped_ROI[:, :, i]
    result[:, :, 3] = cropped_mask  # Set alpha channel based on the mask

    # Set the background (where the mask is zero) to transparent
    result[result[:, :, 3] == 0] = [0, 0, 0, 0]

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.withdraw()
    max_width = screen_width // 2
    max_height = screen_height // 2
    res = resizeImage(result, max_width, max_height)
    return res

def matToHSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    return hsv, hue, saturation, value


def compute_histogram(hue, saturation, bins=10, sat_threshold=0.3):
    histogram = np.zeros(bins, dtype=int)
    bin_size = 360 // bins
    for i in range(hue.shape[0]):
        for j in range(hue.shape[1]):
            if saturation[i, j] > sat_threshold:
                bin_index = int(hue[i, j] // bin_size)
                if bin_index >= bins:
                    bin_index = bins - 1
                histogram[bin_index] += 1
    return histogram


def identify_color(hue, saturation, value, sat_threshold=0.12):
    if saturation < sat_threshold:
        if value > 200:
            return 'White'
        elif value < 10:
            return 'Black'
        else:
            return 'Gray'
    else:
        if 0 <= hue < 15 or 345 <= hue <= 360:
            return 'Red'
        elif 15 <= hue < 45:
            return 'Orange'
        elif 45 <= hue < 75:
            return 'Yellow'
        elif 75 <= hue < 165:
            return 'Green'
        elif 165 <= hue < 260:
            return 'Blue'
        elif 260 <= hue < 290:
            return 'Indigo/Violet'
        elif 290 <= hue < 345:
            return 'Pink'
    return 'Unknown'


def plotHueHistogram(histogram, hue_values, saturation_values, bins=10, sat_threshold=0.2):
    bin_edges = np.linspace(0, 360, bins + 1)
    bin_size = 360 // bins
    low_sat_histogram = np.zeros(bins, dtype=int)
    high_sat_histogram = np.zeros(bins, dtype=int)

    for i in range(len(hue_values)):
        hue = hue_values[i]
        sat = saturation_values[i]
        bin_index = int(hue // bin_size)
        if bin_index >= bins:
            bin_index = bins - 1
        if sat < sat_threshold:
            low_sat_histogram[bin_index] += 1
        else:
            high_sat_histogram[bin_index] += 1

    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    plt.bar(np.arange(bins) - bar_width / 2, low_sat_histogram, width=bar_width, edgecolor='black',
            label='Saturation < 0.3', color='blue')
    plt.bar(np.arange(bins) + bar_width / 2, high_sat_histogram, width=bar_width, edgecolor='black',
            label='Saturation >= 0.3', color='green')

    plt.xticks(range(bins), [f'{int(bin_edges[i])}-{int(bin_edges[i + 1])}' for i in range(bins)])
    plt.xlabel('Hue Range')
    plt.ylabel('Frequency')
    plt.title('Hue Histogram')
    plt.legend()
    plt.show()


def openFileDlg():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def calculate_color_percentages(hue_values, saturation_values, value_values, sat_threshold=0.3):
    color_counts = {}
    total_pixels = len(hue_values)

    for hue, sat, val in zip(hue_values, saturation_values, value_values):
        color = identify_color(hue, sat, val, sat_threshold)
        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1

    color_percentages = {color: (count / total_pixels) * 100 for color, count in color_counts.items()}
    return color_percentages


if __name__ == "__main__":
    img_path = openFileDlg()
    if img_path:
        img = cv2.imread(img_path)
        if img is not None:
            result = remove_background(img)
            cv2.imwrite('cropped_contour_transparent.png', result)

            hsvImg, hue, saturation, value = matToHSV(result)
            hue_values = hue.flatten()
            saturation_values = saturation.flatten()
            value_values = value.flatten()
            histogram = compute_histogram(hue, saturation)

            plotHueHistogram(histogram, hue_values, saturation_values)

            color_percentages = calculate_color_percentages(hue_values, saturation_values, value_values)
            print("Color Percentages:")
            # Sort color percentages in descending order
            sorted_percentages = sorted(color_percentages.items(), key=lambda x: x[1], reverse=True)
            for color, percentage in sorted_percentages:
                print(f"{color}: {percentage:.2f}%")

            # cv2.imshow('Cropped Contour Transparent', result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print("Image not found!")
    else:
        print("No file selected!")
