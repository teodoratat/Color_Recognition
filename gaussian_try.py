import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import tkinter as tk
from tkinter import filedialog


def openFileDlg():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def toHSV(image):
    height, width, _ = image.shape

    # Initialize matrices for H, S, and V components
    Hue = np.zeros((height, width), dtype=np.uint8)
    Sat = np.zeros((height, width), dtype=np.uint8)
    Val = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            B, G, R = image[i, j]

            r = R / 255.0
            g = G / 255.0
            b = B / 255.0

            M = max(r, g, b)
            m = min(r, g, b)
            C = M - m

            V = M
            if V != 0:
                S = C / V
            else:
                S = 0

            H = 0
            if C != 0:
                if M == r:
                    H = 60 * ((g - b) / C % 6)
                elif M == g:
                    H = 60 * ((b - r) / C + 2)
                elif M == b:
                    H = 60 * ((r - g) / C + 4)
            else:
                H = 0

            if H < 0:
                H += 360

            Hue[i, j] = int(H * 255 / 360)
            Sat[i, j] = int(S * 255)
            Val[i, j] = int(V * 255)

    return Hue, Sat, Val


def detect_predominant_colors(image, num_colors=3):
    # Convertirea imaginii în spațiul de culoare HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    Hue, Sat, Val = toHSV(image)

    # Verificarea colțurilor pentru a ignora pixelii negri
    height, width, _ = hsv_image.shape
    corners = [hsv_image[0, 0], hsv_image[0, width - 1], hsv_image[height - 1, 0], hsv_image[height - 1, width - 1]]

    all_black_corners = all((corner[1] == 0 and corner[2] == 0) for corner in corners)
    if all_black_corners:
        return None, None, Hue, Sat, Val

    # Redimensionarea imaginii pentru a lucra mai eficient
    resized_image = cv2.resize(hsv_image, (100, 100), interpolation=cv2.INTER_AREA)

    # Ignorarea pixelilor negri (pixeli cu valoarea de saturație S = 0 sau valoarea V = 0)
    non_black_pixels = resized_image[(resized_image[:, :, 1] != 0) & (resized_image[:, :, 2] != 0)]

    if len(non_black_pixels) == 0:
        print("No significant colors found in the image.")
        return None, None, Hue, Sat, Val

    # Extrage valorile pixelilor de culoare (doar nuanța - Hue)
    hue_values = non_black_pixels[:, 0].reshape(-1, 1)

    # Inițializează modelul GMM
    gmm = GaussianMixture(n_components=num_colors, random_state=42)

    # Antrenează modelul GMM pe nuanță
    gmm.fit(hue_values)
    predominant_colors = gmm.means_

    # Prezicerea culorilor pentru fiecare pixel
    labels = gmm.predict(hue_values)

    # Calcularea procentajelor fiecărei culori predominante
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(hue_values) * 100

    return predominant_colors, percentages, Hue, Sat, Val


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


if __name__ == "__main__":
    img_path = openFileDlg()
    if img_path:
        img = cv2.imread(img_path)
        if img is not None:
            predominant_colors, percentages, Hue, Sat, Val = detect_predominant_colors(img)
            if predominant_colors is not None:
                print("Predominant colors (Hue):")
                for color, percentage in zip(predominant_colors, percentages):
                    if percentage > 25:
                        hue = color[0] * 2  # Convert back to 0-360 range
                        identified_color = identify_color(hue, 1, 1)  # Use dummy saturation and value
                        print(f"Hue: {hue:.2f}, Percentage: {percentage:.2f}%, Color: {identified_color}")

                # Afișare componente HSV
                # cv2.imshow("Initial Image", img)
                # cv2.imshow("Hue", Hue)
                # cv2.imshow("Saturation", Sat)
                # cv2.imshow("Value", Val)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                print("No significant colors found in the image.")
