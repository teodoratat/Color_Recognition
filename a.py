import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

from matplotlib import pyplot as plt


def matToHSV(img):
    height, width = img.shape[:2]
    hsvImg = np.zeros((height, width, 3), dtype=np.uint8)
    hue = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            pixel = img[i, j]
            B, G, R = pixel[0], pixel[1], pixel[2]
            r, g, b = R / 255.0, G / 255.0, B / 255.0
            M = max(r, g, b)
            m = min(r, g, b)
            C = M - m
            V = M
            S = C / V if V != 0 else 0
            H = 0
            hue = np.zeros((height, width), dtype=np.float32)
            saturation = np.zeros((height, width), dtype=np.float32)
            if C != 0:
                if M == r:
                    H = 60 * (g - b) / C
                elif M == g:
                    H = 120 + 60 * (b - r) / C
                elif M == b:
                    H = 240 + 60 * (r - g) / C

            if H < 0:
                H += 360

            hue[i, j] = H
            saturation[i, j] = S
            hsvImg[i, j] = [H, S * 255, V * 255]  # Adjust saturation value range

    return hsvImg, hue


# Mouse callback function
def MyCallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param
        # Do something with the image and the coordinates (x, y)
        print(f"Pos(x,y): {x},{y}  Color(RGB): {img[y, x][2]},{img[y, x][1]},{img[y, x][0]}")


def openFileDlg():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    return file_path


def checkBackground(file_path):
    img = cv2.imread(file_path)
    if img is None:
        return None

    height, width = img.shape[:2]

    # Define corner coordinates
    corners = [
        (0, 0),  # Top-left
        (0, width - 1),  # Top-right
        (height - 1, 0),  # Bottom-left
        (height - 1, width - 1)  # Bottom-right
    ]

    # Extract corner pixel values
    corner_colors = [img[y, x] for y, x in corners]

    # Check if all corners have the same color
    background_color = corner_colors[0]
    for color in corner_colors:
        if not np.array_equal(background_color, color):
            return False

    return True


def resizeImage(img, max_width, max_height):
    height, width = img.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
    return img


def testMouseClick():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.withdraw()

    max_width = screen_width // 2  # A quarter of the screen width
    max_height = screen_height // 2  # A quarter of the screen height

    while True:
        fname = openFileDlg()
        if not fname:
            break  # Exit if no file is selected
        src = cv2.imread(fname)
        if src is None:
            print(f"Failed to load image: {fname}")
            continue

        # Resize the image
        src = resizeImage(src, max_width, max_height)

        # Create a window
        cv2.namedWindow("My Window", 1)

        # Set the callback function for any mouse event
        cv2.setMouseCallback("My Window", MyCallBackFunc, src)

        # Show the HSV image
        cv2.imshow("My Window", src)

        # Wait until user presses some key
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break

        cv2.destroyWindow("My Window")

def plotHueHistogram(hue):
    hue_flat = hue.flatten()
    plt.hist(hue_flat, bins=10, range=(0, 360), color='blue', alpha=0.7)
    plt.title('Hue Histogram')
    plt.xlabel('Hue')
    plt.ylabel('Frequency')
    plt.show()


# def testMouseClick1():
#     root = tk.Tk()
#     screen_width = root.winfo_screenwidth()
#     screen_height = root.winfo_screenheight()
#     root.withdraw()
#
#     max_width = screen_width // 2  # A quarter of the screen width
#     max_height = screen_height // 2  # A quarter of the screen height
#
#     while True:
#         fname = openFileDlg()
#         if not fname:
#             break  # Exit if no file is selected
#         src = cv2.imread(fname)
#         if src is None:
#             print(f"Failed to load image: {fname}")
#             continue
#
#         # Resize the image
#         src = resizeImage(src, max_width, max_height)
#
#         # Convert to HSV and extract hue
#         hsvImg, hue = matToHSV(src)
#
#         # Plot the hue histogram
#         plotHueHistogram(hue)
#
#         # Wait until user presses some key
#         key = cv2.waitKey(0)
#         if key == 27:  # ESC key to exit
#             break
#
#         cv2.destroyAllWindows()


if __name__ == "__main__":
    testMouseClick()
