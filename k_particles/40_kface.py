"""Implement Conway's game of life."""

import subprocess
import numpy as np
import nibabel as nb
import scipy as sp
import cv2
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
FILE2 = "/Users/faruk/Documents/temp-k_particles/pngs/kevin-01.png"

OUTDIR = "/Users/faruk/Documents/temp-k_particles/40_kface"

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
kspace2 = np.fft.fftshift(np.fft.fft2(ispace2))
print(ispace2.shape)

# Interpolate
betas = np.linspace(0 + 0j, 1 + 1j, NR_FRAMES)
for i in range(NR_FRAMES):
    kspace_temp = kspace1 * (1+1j - betas[i]) + kspace2 * betas[i]
    recon = np.fft.ifft2(np.fft.ifftshift(kspace_temp))

    # Save
    img1 = normalize_to_uint8(np.log10(np.abs(kspace_temp) + 1), perc_min=0.1, perc_max=99.9)
    img2 = normalize_to_uint8(np.abs(recon), perc_min=0.1, perc_max=99.9)
    save_frame_as_png_2(img1, img2, OUTDIR, i)

# =============================================================================
command = "ffmpeg -y "
command += "-framerate {} ".format(int(FRAMERATE))
command += "-i {}/frame-%04d.png ".format(OUTDIR)
command += "-c:v libx264 -r 30 -pix_fmt yuv420p "
command += "{}/00_movie.mp4".format(OUTDIR)

# Execute command
print("==========!!!!!!!!!!!!==========")
print(command)
print("==========!!!!!!!!!!!!==========")
subprocess.run(command, shell=True, check=True)


print("Finished.")
