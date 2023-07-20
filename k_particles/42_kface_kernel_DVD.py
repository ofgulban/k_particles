"""Implement Conway's game of life."""

import subprocess
import numpy as np
import nibabel as nb
import scipy as sp
import cv2
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
FILE2 = "/Users/faruk/Documents/temp-k_particles/pngs/kernel-kevin.png"

OUTDIR = "/Users/faruk/Documents/temp-k_particles/42_kface_kernel_DVD"

NR_FRAMES = 300
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

# Move kernel in kspace
roll1 = random.choice([-1, 1]) * random.choice([1, 2])
roll2 = random.choice([-1, 1]) * random.choice([1, 2])
for i in range(NR_FRAMES):
    ispace2 = np.roll(ispace2, roll1, axis=0)
    ispace2 = np.roll(ispace2, roll2, axis=1)

    kspace_temp = np.copy(kspace1)

    # Real imaginary to magnitude and phase
    kspace_mag = np.abs(kspace_temp)
    kspace_pha = np.angle(kspace_temp)

    # Mask magnitude
    kspace_mag *= ispace2

    # Calculate real and imaginary parts
    kspace_real = kspace_mag * np.cos(kspace_pha)
    kspace_imag = kspace_mag * np.sin(kspace_pha)

    # Create complex numbers using real and imaginary parts
    kspace_temp = kspace_real + 1j * kspace_imag
    recon = np.fft.ifft2(np.fft.ifftshift(kspace_temp))

    img1 = normalize_to_uint8(np.log10(np.abs(kspace_temp) + 1), perc_min=0.1, perc_max=99.9)
    img2 = normalize_to_uint8(np.abs(recon), perc_min=0.1, perc_max=99.9)
    save_frame_as_png_2(img1, img2, OUTDIR, i=i)

    # Edge conditions / bounce
    collapse1 = np.sum(ispace2, axis=0)
    collapse2 = np.sum(ispace2, axis=1)

    # if collapse1[0] > 0:
    #     roll2 *= -1
    # elif collapse1[-1] > 0:
    #     roll2 *= -1
    # if collapse2[0] > 0:
    #     roll1 *= -1
    # elif collapse2[-1] > 0:
    #     roll1 *= -1

    if collapse1[0] > 0:
        if roll2 > 1:
            roll2 *= -1
        else:
            roll2 *= -random.choice([1, 2])
    elif collapse1[-1] > 0:
        if roll2 > 1:
            roll2 *= -1
        else:
            roll2 *= -random.choice([1, 2])
    if collapse2[0] > 0:
        if roll2 > 1:
            roll1 *= -1
        else:
            roll1 *= -random.choice([1, 2])
    elif collapse2[-1] > 0:
        if roll2 > 1:
            roll1 *= -1
        else:
            roll1 *= -random.choice([1, 2])

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
