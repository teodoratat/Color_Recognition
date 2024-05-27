import numpy as np
from PIL import Image


def get_new_val(old_val, nc):
    """
    Get the "closest" colour to old_val in the range [0,1] per channel divided
    into nc values.
    """
    return np.round(old_val * (nc - 1)) / (nc - 1)


def fs_dither(img, nc):
    """
    Floyd-Steinberg dither the image img into a palette with nc colours per
    channel.
    """
    arr = np.array(img, dtype=float) / 255
    new_arr = np.zeros_like(arr)

    height, width, _ = arr.shape

    for ir in range(height):
        for ic in range(width):
            old_val = arr[ir, ic].copy()
            new_val = get_new_val(old_val, nc)
            new_arr[ir, ic] = new_val
            err = old_val - new_val
            if ic < width - 1:
                arr[ir, ic + 1] += err * 7 / 16
            if ir < height - 1:
                if ic > 0:
                    arr[ir + 1, ic - 1] += err * 3 / 16
                arr[ir + 1, ic] += err * 5 / 16
                if ic < width - 1:
                    arr[ir + 1, ic + 1] += err / 16

    carr = np.array(new_arr * 255, dtype=np.uint8)
    return Image.fromarray(carr)



def palette_reduce(img, nc):
    """Simple palette reduction without dithering."""
    arr = np.array(img, dtype=float) / 255
    arr = get_new_val(arr, nc)
    carr = np.array(arr / np.max(arr) * 255, dtype=np.uint8)
    return Image.fromarray(carr)


# Load the image
img_name = './images/green_dress.jpg'
img = Image.open(img_name)

# Parameters
nc_values = (2, 3, 4, 8, 16)

# Perform dithering and palette reduction for each nc value
for nc in nc_values:
    print('nc =', nc)

    print("aaaa")
    # Perform Floyd-Steinberg dithering
    print (2)
    dithered_img = fs_dither(img, nc)
    print (3)
    dithered_img.save(f'dimg-{nc}.jpg')

    # Perform palette reduction
    reduced_img = palette_reduce(img, nc)
    reduced_img.save(f'rimg-{nc}.jpg')
