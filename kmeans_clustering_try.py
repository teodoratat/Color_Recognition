import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog
from collections import Counter
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2, d):
    sum_val = 0.0
    for i in range(d):
        sum_val += (p1[i] - p2[i]) ** 2
    return np.sqrt(sum_val)

def most_frequent_color(src):
    data = src.reshape(-1, 3)
    data = [tuple(row) for row in data]
    count = Counter(data)
    return count.most_common(1)[0][0]

def assign_pixels_to_centroids(src, m, k, height, width, background_color):
    l = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            if tuple(src[i, j]) == background_color:
                l[i][j] = -1  # Mark background pixels with -1
                continue

            min_distance = float('inf')
            nearest_centroid = 0

            for t in range(k):
                bbb = m[t][3:6]
                distance = euclidean_distance(src[i, j], bbb, 3)
                if distance < min_distance:
                    min_distance = distance
                    nearest_centroid = t

            l[i][j] = nearest_centroid
    return l

def assign_colors(k, src_color, l):
    cluster_colors = []
    cluster_counts = np.zeros(k, dtype=int)
    for cluster_idx in range(k):
        h = np.zeros((256, 256, 256), dtype=int)

        for i in range(src_color.shape[0]):
            for j in range(src_color.shape[1]):
                if l[i][j] == cluster_idx:
                    b, g, r = src_color[i, j]
                    h[b, g, r] += 1
                    cluster_counts[cluster_idx] += 1

        max_count = 0
        dominant_color = None

        for b in range(256):
            for g in range(256):
                for r in range(256):
                    if h[b, g, r] > max_count:
                        max_count = h[b, g, r]
                        dominant_color = [b, g, r]

        cluster_colors.append(dominant_color)

    total_pixels = src_color.shape[0] * src_color.shape[1]
    color_percentages = (cluster_counts / total_pixels) * 100

    color_info = [(cluster_colors[i], color_percentages[i]) for i in range(k) if cluster_colors[i] is not None]
    color_info.sort(key=lambda x: x[1], reverse=True)

    for color, percentage in color_info:
        print(f"Color {color}: {percentage:.2f}%")

    # Plot histogram
    plot_color_histogram(color_info)

    return cluster_colors

def plot_color_histogram(color_info):
    colors = [color for color, _ in color_info]
    percentages = [percentage for _, percentage in color_info]

    # Convert BGR to RGB for matplotlib display
    colors_rgb = [np.array(color)[::-1] for color in colors]

    fig, ax = plt.subplots()
    bars = ax.bar(range(len(colors_rgb)), percentages, color=np.array(colors_rgb)/255)

    ax.set_xlabel('Colors')
    ax.set_ylabel('Percentage')
    ax.set_title('Color Percentages')

    # Set x-ticks to be color bars
    ax.set_xticks(range(len(colors_rgb)))
    ax.set_xticklabels([f'Color {i+1}' for i in range(len(colors_rgb))])

    plt.show()

def kmeans_clustering(k):
    colors = [np.random.randint(0, 256, 3) for _ in range(k)]
    color_bw = [np.random.randint(0, 256) for _ in range(k)]

    fname = open_file_dlg()
    if fname:
        src = cv2.imread(fname, cv2.IMREAD_COLOR)
        src = cv2.resize(src, (src.shape[1] // 4, src.shape[0] // 4))  # Resize image to quarter its original size
        height, width = src.shape[:2]

        background_color = most_frequent_color(src)
        print(f"Background color: {background_color}")

        centers = np.zeros((k, 6), dtype=int)

        for i in range(k):
            rand_i = random.randint(0, height - 1)
            rand_j = random.randint(0, width - 1)
            while np.all(src[rand_i, rand_j] == background_color):
                rand_i = random.randint(0, height - 1)
                rand_j = random.randint(0, width - 1)
            centers[i][0] = rand_i
            centers[i][1] = rand_j
            centers[i][2] = src[rand_i, rand_j][0]
            centers[i][3] = src[rand_i, rand_j][1]
            centers[i][4] = src[rand_i, rand_j][2]

        l = assign_pixels_to_centroids(src, centers, k, height, width, background_color)

        arrays_equal = False
        kontor = 0

        while not arrays_equal and kontor < 100:
            kontor += 1
            new_l = assign_pixels_to_centroids(src, centers, k, height, width, background_color)
            arrays_equal = np.array_equal(l, new_l)
            l = new_l

        colors = assign_colors(k, src, l)

        dst = np.zeros_like(src)
        dst_bw = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                if l[i][j] == -1:
                    dst[i, j] = background_color
                    dst_bw[i, j] = 255
                else:
                    dst[i, j] = colors[l[i][j]]
                    dst_bw[i, j] = color_bw[l[i][j]]

        cv2.imshow("src", src)
        cv2.imshow("dst", dst)
        cv2.imshow("dst black white", dst_bw)
        cv2.waitKey()

def open_file_dlg():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

# Call kmeans_clustering function with desired value of k
kmeans_clustering(5)  # Example with k=5
