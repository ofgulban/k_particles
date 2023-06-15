"""Convolution in image space is multiplication in k space."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"

OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii-05_mask_walk"

NR_FRAMES = 300
RADIUS_TRAJ = 120  # TODO normalize this
RADIUS_MASK = 0.05
KAPPA = 0.9
P = 1
# =============================================================================
create_output_directory(OUTDIR)

# Load data
nii1 = nb.load(FILE1)
ispace = np.asarray(nii1.dataobj, dtype=np.float32).T[::-1, :]
dims = ispace.shape
print(dims)

img1 = normalize_to_uint8(ispace, perc_min=1, perc_max=99)
save_frame_as_png_1(img1, OUTDIR, tag="step-01_ispace", zoom_factor=2)

# Fourier transform
kspace = np.fft.fftshift(np.fft.fft2(ispace))
img2 = normalize_to_uint8(np.abs(kspace), perc_min=1, perc_max=99)
save_frame_as_png_1(img2, OUTDIR, tag="step-02_kspace_abs", zoom_factor=2)

coords_walk = generate_points_on_circle(center=(0, 0), radius=RADIUS_TRAJ,
                                        start=0, end=4*np.pi, 
                                        num_points=NR_FRAMES)

mask = np.zeros(dims)
for i in range(NR_FRAMES):
    # Generate mask
    coords_mask = create_2D_array_coordinates(size_x=dims[0], size_y=dims[1], 
                                              center_x=coords_walk[i, 0], 
                                              center_y=coords_walk[i, 1])

    norm = Lp_norm(x1=coords_mask[:, :, 0], x2=coords_mask[:, :, 1], p=P)
    norm /= np.max(norm)
    mask *= KAPPA
    mask[norm < RADIUS_MASK] = 1

    # Mask k space
    temp = kspace * mask

    # Reconstruct image
    recon = np.fft.ifft2(temp)

    # Exports
    img1 = normalize_to_uint8(np.abs(temp), perc_min=1, perc_max=99)
    img2 = normalize_to_uint8(np.abs(recon), perc_min=1, perc_max=99)
    save_frame_as_png_2(img1, img2, OUTDIR, i=i, zoom_factor=2)

outname = "00_movie"
outname += f"_P-{P}"
outname += f"_RadTraj-{RADIUS_TRAJ}"
outname += f"_RadMask-{RADIUS_MASK}"
outname += f"_Kappa-{KAPPA}"
outname = outname.replace(".", "pt")
make_movie(OUTDIR, tag=outname, framerate=30)


print("Finished.")