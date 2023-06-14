import os
import numpy as np
from scipy.ndimage import zoom
import cv2


def create_output_directory(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print("\n  Output directory:\n    {}\n".format(outdir))


def save_frame_as_png(img1, img2, outdir, i, zoom_factor=8):
    img_out = np.hstack((img1, img2))
    img_out = zoom(img_out, zoom_factor, order=0)
    outname = os.path.join(outdir, "frame-{}.png".format(str(i).zfill(4)))
    cv2.imwrite(outname, img_out)


def generate_points_on_circle(center, radius, start, end, num_points):
    coords = np.zeros((num_points, 2))
    theta = np.linspace(start, end, num_points, endpoint=False)
    coords[:, 0] = center[0] + radius * np.cos(theta)
    coords[:, 1] = center[1] + radius * np.sin(theta)
    return coords


def gaussian_interpolation(img, x, y, alpha=1.0, hermitian=True):
    # Find the four neighboring grid points
    i = int(np.round(x))
    j = int(np.round(y))

    # Compute distances to each pixel (NOTE: Part of operations is redundant)
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
    img[i-1, j-1] = w00 * alpha
    img[i-1, j] = w01 * alpha
    img[i-1, j+1] = w02 * alpha
    img[i, j-1] = w10 * alpha
    img[i, j] = w11 * alpha
    img[i, j+1] = w12 * alpha
    img[i+1, j-1] = w20 * alpha
    img[i+1, j] = w21 * alpha
    img[i+1, j+1] = w22 * alpha

    # Hermitian symmetry
    if hermitian:
        img += img[::-1, ::-1]

    # img += img[::-1, :]
    # img += img[:, ::-1]

    return img


def coordinates_to_lattice(coord_x, coord_y, lattice, alpha, hermitian=True):
    # Insert points onto a clean lattice
    temp = np.zeros(lattice.shape)
    lattice += gaussian_interpolation(temp, coord_x, coord_y, alpha, hermitian)

    # Clip
    lattice[lattice > 1] = 1
    return lattice


def normalize_to_uint8(array, perc_min=0, perc_max=100):
    val_min, val_max = np.percentile(array, [perc_min, perc_max])
    array[array > val_max] = val_max
    array[array < val_min] = val_min    
    array = (array - val_min) / (val_max - val_min) * 255
    return array.astype(np.uint8)
