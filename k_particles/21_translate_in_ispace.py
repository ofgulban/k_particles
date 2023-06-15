"""Step by step introduction to k space."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii-02_point"


DIMS = [128, 128]
NR_FRAMES = 128

# -----------------------------------------------------------------------------
def create_square(center, radius, dims):
    # Create an empty 2D array
    square = np.zeros(dims)

    # Calculate the square's boundaries
    x_min = int(center[0] - radius)
    x_max = int(center[0] + radius)
    y_min = int(center[1] - radius)
    y_max = int(center[1] + radius)

    # Check boundary
    if x_min < 0:
        x_min = 0
    if x_max >= dims[0]:
        x_max = dims[0]-1
    if y_min < 0:
        y_min = 0
    if y_max >= dims[1]:
        y_max = dims[1]-1

    # Set the values within the square boundaries to 1
    square[x_min:x_max+1, y_min:y_max+1] = 1

    return square


def create_trajectory_linear(start, end, nr_points):
    coords = np.zeros((nr_points, 2))
    coords[:, 0] = np.linspace(start[0], end[0], nr_points, endpoint=True)
    coords[:, 1] = np.linspace(start[1], end[1], nr_points, endpoint=True)
    return coords


# -----------------------------------------------------------------------------
# # make gaussian
# def create_2d_gaussian(size, sigma):
#     # Calculate the center point of the array
#     center = size // 2

#     # Create a coordinate grid
#     x, y = np.meshgrid(np.arange(size), np.arange(size))

#     # Calculate the Gaussian values
#     gaussian = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))

#     return gaussian

# sigma = 10
# orig = create_2d_gaussian(DIMS[0], sigma)
# -----------------------------------------------------------------------------
walk_coords = create_trajectory_linear(start=(DIMS[0]//2, 0), 
                                       end=(DIMS[0]//2, DIMS[1]-1),
                                       nr_points=NR_FRAMES)

orig = create_square(center=(64, 64), radius=10, dims=DIMS)
print(np.max(orig))

for i in range(NR_FRAMES):

    fourier_space = np.fft.fftshift(np.fft.fft2(orig))

    img1 = normalize_to_uint8(orig, perc_min=0, perc_max=100)
    img2 = normalize_to_uint8(np.log10(np.abs(fourier_space) + 1), 
                              perc_min=0, perc_max=100)
    img3 = normalize_to_uint8(np.abs(np.angle(fourier_space)), 
                              perc_min=0, perc_max=100)

    # Save
    img_out = np.hstack((img1, img2, img3)) 
    img_out = zoom(img_out, 4, order=0)
    outname = os.path.join(OUTDIR, "frame-{}.png".format(str(i).zfill(4)))
    cv2.imwrite(outname, img_out)

print("Finished.")