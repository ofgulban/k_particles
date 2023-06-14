"""Step by step introduction to k space."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii-03_point"


DIMS = [128, 128]
NR_FRAMES = 30


for i, j in enumerate(range(5, 5+NR_FRAMES)):
    orig = np.zeros(DIMS)
    orig[64-j:64+j, 64-j:64+j] = 1

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