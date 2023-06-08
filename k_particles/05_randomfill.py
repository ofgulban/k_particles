import os
import numpy as np
import cv2
from scipy.ndimage import zoom
import random

OUTDIR = "/home/faruk/Documents/temp-k_particles/05_randomfill"
DIMS = [512, 512]

NUM_POINTS = DIMS[0] * DIMS[1] // 32  # Number of points to generate

# =============================================================================


def generate_random_points(dims, nr_points):
    coords = np.zeros((nr_points, 2))
    print(coords.shape)
    for i in range(nr_points):
        coords[i, 0] = random.uniform(1, dims[0]-2)
        coords[i, 1] = random.uniform(1, dims[1]-2)
    return coords


def gaussian_interpolation(img, x, y):
    # Find the four neighboring grid points
    i = int(np.round(x))
    j = int(np.round(y))

    p00 = abs(x - (i-1)), abs(y - (j-1))
    p01 = abs(x - (i-1)), abs(y - j)
    p02 = abs(x - (i-1)), abs(y - (j+1))

    p10 = abs(x - i), abs(y - (j-1))
    p11 = abs(x - i), abs(y - j)
    p12 = abs(x - i), abs(y - (j+1))

    p20 = abs(x - (i+1)), abs(y - (j-1))
    p21 = abs(x - (i+1)), abs(y - j)
    p22 = abs(x - (i+1)), abs(y - (j+1))

    # Calculate the weights
    w00 = np.exp(-p00[0]**2 - p00[1]**2)
    w01 = np.exp(-p01[0]**2 - p01[1]**2)
    w02 = np.exp(-p02[0]**2 - p02[1]**2)

    w10 = np.exp(-p10[0]**2 - p10[1]**2)
    w11 = np.exp(-p11[0]**2 - p11[1]**2)
    w12 = np.exp(-p12[0]**2 - p12[1]**2)

    w20 = np.exp(-p20[0]**2 - p20[1]**2)
    w21 = np.exp(-p21[0]**2 - p21[1]**2)
    w22 = np.exp(-p22[0]**2 - p22[1]**2)

    # Perform bilinear interpolation
    img[i-1, j-1] = w00
    img[i-1, j] = w01
    img[i-1, j+1] = w02
    img[i, j-1] = w10
    img[i, j] = w11
    img[i, j+1] = w12
    img[i+1, j-1] = w20
    img[i+1, j] = w21
    img[i+1, j+1] = w22

    return img


coords = generate_random_points(DIMS, NUM_POINTS)
kspace = np.zeros(DIMS, dtype=complex)
for i in range(NUM_POINTS):
    x, y = coords[i, 0], coords[i, 1]

    # Insert particle
    kspace.real = gaussian_interpolation(kspace.real, x, y)

# Clip some part of k space
kspace.real[0:128, :] = 0
kspace.real[384:, :] = 0
kspace.real[:, 0:128] = 0
kspace.real[:, 384:] = 0

# Convert to image format
img1 = np.abs(kspace)
img1 = img1 / np.max(img1) * 255

# Reconstruct image
ispace = np.fft.ifft2(kspace)

# Convert to image format
img2 = np.abs(ispace)
img2[img2 > np.percentile(img2, 95)] = np.percentile(img2, 95)
img2 = img2 / np.max(img2) * 255
img2 = np.fft.ifftshift(img2)  # TODO: Figure out why I need this

# Save
img = np.hstack((img1, img2))
img = zoom(img, 2, order=0)
outname = os.path.join(OUTDIR, "test.png")
cv2.imwrite(outname, img.astype(np.uint8))

print("Finished.")
