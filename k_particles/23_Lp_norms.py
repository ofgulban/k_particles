"""Step by step introduction to k space."""
import os
import subprocess
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii-04_Lp_norms"

NR_FRAMES = 300
FRAMERATE = 30

# -----------------------------------------------------------------------------
# Create a 2D array with coordinates centered at (0, 0)
size = 128
x = np.arange(-size // 2, size // 2 + 1)
y = np.arange(-size // 2, size // 2 + 1)
coords_lattice = np.transpose(np.asarray(np.meshgrid(x, y)), (1, 2, 0))

nr_pixels = coords_lattice.shape[0] * coords_lattice.shape[1]
radius = 48

for i, j in enumerate(np.linspace(start=0.01, stop=2.00, num=NR_FRAMES, endpoint=True)):
    norm = Lp_norm(coords_lattice[:, :, 0], coords_lattice[:, :, 1], j)
    ispace = np.ones(norm.shape)
    ispace[norm >= radius] = 0
    ispace[norm < radius-5] = 0
    kspace = np.fft.fftshift(np.fft.fft2(ispace))

    img1 = normalize_to_uint8(ispace, perc_min=0, perc_max=100)
    img2 = normalize_to_uint8(np.log10(np.abs(kspace) + 1), perc_min=0, perc_max=100)
    img3 = normalize_to_uint8(np.abs(np.angle(kspace)), perc_min=0, perc_max=100)

    # Save
    img_out = np.hstack((img1, img2, img3)) 
    img_out = zoom(img_out, 4, order=0)
    outname = os.path.join(OUTDIR, "frame-{}.png".format(str(i).zfill(4)))
    cv2.imwrite(outname, img_out)


# =============================================================================
command = "ffmpeg "
command += "-y -framerate {} ".format(FRAMERATE)
command += "-i {}/frame-%04d.png ".format(OUTDIR)
command += "-c:v libx264 -pix_fmt yuv420p "
command += "{}/00_movie.mp4".format(OUTDIR)

# Execute command
print(command)
subprocess.run(command, shell=True)

print("Finished.")