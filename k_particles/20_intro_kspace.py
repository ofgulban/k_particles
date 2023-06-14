"""Step by step introduction to k space."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii-02_point"


DIMS = [128, 128]

orig = np.zeros(DIMS)
orig[64-10:64+10, 64-10:64+10] = 1

# Save
img1 = orig * 255
img1 = zoom(img1, 4, order=0)
outname = os.path.join(OUTDIR, "frame1-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img1)
print(img1.shape)


fourier_space = np.fft.fftshift(np.fft.fft2(orig))
img2 = normalize_to_uint8(fourier_space.real, perc_min=0, perc_max=100)
img3 = normalize_to_uint8(fourier_space.imag, perc_min=0, perc_max=100)

img2 = zoom(img2, 4, order=0)
outname = os.path.join(OUTDIR, "frame2_real-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img2)

img3 = zoom(img3, 4, order=0)
outname = os.path.join(OUTDIR, "frame3_imag-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img3)


img4 = normalize_to_uint8(np.abs(fourier_space), perc_min=0, perc_max=100)
img5 = normalize_to_uint8(np.angle(fourier_space), perc_min=0, perc_max=100)

img4 = zoom(img4, 4, order=0)
outname = os.path.join(OUTDIR, "frame4_mag-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img4)

img5 = zoom(img5, 4, order=0)
outname = os.path.join(OUTDIR, "frame5_phase-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img5)

recon_space = np.fft.ifft2(fourier_space)


img6 = recon_space.real
img7 = recon_space.imag

img6 = normalize_to_uint8(recon_space.real, perc_min=0, perc_max=100)
img7 = normalize_to_uint8(recon_space.imag, perc_min=0, perc_max=100)

img6 = zoom(img6, 4, order=0)
outname = os.path.join(OUTDIR, "frame6_ifft_real-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img6)

img7 = zoom(img7, 4, order=0)
outname = os.path.join(OUTDIR, "frame7_ifft_imag-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img7)


img8 = np.abs(recon_space)
img9 = np.angle(recon_space)


print(np.max(img8) - np.min(img8))
print(np.max(img9) - np.min(img9))

img8 = normalize_to_uint8(img8, perc_min=0, perc_max=100)
img9 = normalize_to_uint8(img9, perc_min=0, perc_max=100)

img8 = zoom(img8, 4, order=0)
outname = os.path.join(OUTDIR, "frame8_ifft_mag-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img8)

img9 = zoom(img9, 4, order=0)
outname = os.path.join(OUTDIR, "frame9_ifft_phase-{}.png".format(str(1).zfill(4)))
cv2.imwrite(outname, img9)


print("Finished.")