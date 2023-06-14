"""Step by step introduction to k space."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii-02_point"


DIMS = [128, 128]
NR_FRAMES = 128 * 2

# -----------------------------------------------------------------------------
# # make square
# orig = np.zeros(DIMS)
# orig[64-5:64+5, 64-5:64+5] = 1

# -----------------------------------------------------------------------------
# make gaussian
def create_2d_gaussian(size, sigma):
    # Calculate the center point of the array
    center = size // 2

    # Create a coordinate grid
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    # Calculate the Gaussian values
    gaussian = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))

    return gaussian

sigma = 10
orig = create_2d_gaussian(DIMS[0], sigma)
# -----------------------------------------------------------------------------


for i in range(NR_FRAMES):
    orig = np.roll(orig, 1, axis=1)
    img1 = normalize_to_uint8(orig, perc_min=0, perc_max=100)

    fourier_space = np.fft.fftshift(np.fft.fft2(orig))
    img2 = normalize_to_uint8(np.log10(np.abs(fourier_space) + 1), perc_min=0, perc_max=100)
    img3 = normalize_to_uint8(np.abs(np.angle(fourier_space)), perc_min=0, perc_max=100)

    # Save
    img_out = np.hstack((img1, img2, img3)) 
    img_out = zoom(img_out, 4, order=0)
    outname = os.path.join(OUTDIR, "frame-{}.png".format(str(i).zfill(4)))
    cv2.imwrite(outname, img_out)

print("Finished.")