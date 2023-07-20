"""Implement Conway's game of life."""

import subprocess
import numpy as np
import nibabel as nb
import scipy as sp
import cv2
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
FILE2 = "/Users/faruk/Documents/temp-k_particles/pngs/kernel-kevin.png"

OUTDIR = "/Users/faruk/Documents/temp-k_particles/41_kface_kernel"

NR_FRAMES = 100
FRAMERATE = 30

# =============================================================================
# Load brain data
nii1 = nb.load(FILE1)
ispace1 = np.asarray(nii1.dataobj, dtype=np.float32).T[::-1, :]
ispace1 = sp.ndimage.zoom(ispace1, 0.5)
ispace1 = ispace1 / np.max(ispace1)
kspace1 = np.fft.fftshift(np.fft.fft2(ispace1))
print(ispace1.shape)

# Read the PNG image in grayscale mode
ispace2 = cv2.imread(FILE2, cv2.IMREAD_GRAYSCALE)
ispace2 = sp.ndimage.zoom(ispace2, 0.5)
ispace2 = ispace2 / np.max(ispace2)

kspace1 *= ispace2
recon = np.fft.ifft2(np.fft.ifftshift(kspace1))

img1 = normalize_to_uint8(np.log10(np.abs(kspace1) + 1), perc_min=0.1, perc_max=99.9)
img2 = normalize_to_uint8(np.abs(recon), perc_min=0.1, perc_max=99.9)
save_frame_as_png_2(img1, img2, OUTDIR, i=0)

print("Finished.")
