import cv2
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt


def region_growing(image, seed_point, threshold):
    h, w = image.shape[:2]
    labels = np.zeros((h, w), np.int32)
    label = 1

    image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]

    queue = deque([seed_point])
    seed_hue = hue_channel[seed_point[1], seed_point[0]]
    region_hue_avg = seed_hue
    n = 1

    while queue:
        x, y = queue.popleft()
        labels[y, x] = label

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and labels[ny, nx] == 0:
                neighbor_hue = hue_channel[ny, nx]
                hue_diff = min(abs(int(neighbor_hue) - int(region_hue_avg)),
                               180 - abs(int(neighbor_hue) - int(region_hue_avg)))
                if hue_diff < threshold:
                    queue.append((nx, ny))
                    labels[ny, nx] = label
                    region_hue_avg = (region_hue_avg * n + neighbor_hue) / (n + 1)
                    n += 1

    return labels


def find_predominant_colors(image, labels):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    predominant_colors = []
    color_sizes = []

    for label in np.unique(labels):
        if label == 0:
            continue
        mask = (labels == label).astype(np.uint8)
        masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        hue_values = masked_hsv[:, :, 0].flatten()
        hue_values = hue_values[hue_values > 0]
        if hue_values.size == 0:
            continue
        hist = cv2.calcHist([hue_values], [0], None, [180], [0, 180])
        predominant_hue_bin = np.argmax(hist)
        predominant_color = np.array([predominant_hue_bin, 255, 255], dtype=np.uint8)
        predominant_color_rgb = cv2.cvtColor(np.array([[predominant_color]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0]
        predominant_colors.append(predominant_color_rgb)
        color_sizes.append(np.sum(mask))

    total_size = np.sum(color_sizes)
    color_percentages = [size / total_size * 100 for size in color_sizes]

    return predominant_colors, color_percentages


def get_random_seed_points(image, num_points=5):
    h, w = image.shape[:2]
    seed_points = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(num_points)]
    return seed_points


def detect_principal_colors(image_path, num_seed_points=5, threshold=10):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Image not found.")
    except Exception as e:
        print("Error:", e)
        return None, None

    seed_points = get_random_seed_points(image, num_seed_points)
    all_labels = np.zeros(image.shape[:2], np.int32)

    for seed_point in seed_points:
        labels = region_growing(image, seed_point, threshold)
        all_labels = np.maximum(all_labels, labels)

    predominant_colors, color_percentages = find_predominant_colors(image, all_labels)
    return predominant_colors, color_percentages


def main():
    image_path = './images/white_shirt.jpg'
    predominant_colors, color_percentages = detect_principal_colors(image_path)

    if predominant_colors is not None and color_percentages is not None:
        print("Predominant colors:", predominant_colors)
        print("Color percentages:", color_percentages)

        color_labels = [f"Color {i + 1}" for i in range(len(predominant_colors))]
        color_values = [tuple(color) for color in predominant_colors]

        fig, ax = plt.subplots()
        ax.pie(color_percentages, labels=color_labels, colors=np.array(predominant_colors) / 255, autopct='%1.1f%%')
        ax.axis('equal')

        # Make background transparent
        fig.patch.set_alpha(0)
        fig.patch.set_facecolor('none')
        ax.patch.set_alpha(0)
        ax.patch.set_facecolor('none')

        plt.show()


if __name__ == "__main__":
    main()
