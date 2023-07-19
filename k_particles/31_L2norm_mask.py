"""Step by step introduction to k space."""
import os
import subprocess
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii-07_L2norm"

NR_FRAMES = 300
FRAMERATE = 30

# -----------------------------------------------------------------------------
# Load data
nii1 = nb.load(FILE1)
ispace = np.asarray(nii1.dataobj, dtype=np.float32).T[::-1, :]
dims = ispace.shape
print(dims)

# Create a 2D array with coordinates centered at (0, 0)
x = np.arange(-dims[0] // 2, dims[0] // 2)
y = np.arange(-dims[1] // 2, dims[1] // 2)
coords_lattice = np.transpose(np.asarray(np.meshgrid(x, y)), (1, 2, 0))
print(coords_lattice.shape)

nr_pixels = coords_lattice.shape[0] * coords_lattice.shape[1]

# Initialize k space
kspace = np.fft.fftshift(np.fft.fft2(ispace))


for i, radius in enumerate(np.geomspace(start=dims[0]//2*np.sqrt(2), stop=1, num=NR_FRAMES, endpoint=True)):
    norm = Lp_norm(coords_lattice[:, :, 0], coords_lattice[:, :, 1], 2)
    idx = norm > radius
    kspace.real[idx] = 0
    kspace.imag[idx] = 0
    recon = np.fft.ifft2(np.fft.ifftshift(kspace))

    # img1 = normalize_to_uint8(ispace, perc_min=1, perc_max=99)
    img2 = normalize_to_uint8(np.log10(np.abs(kspace) + 1), perc_min=0.1, perc_max=99.9)
    img3 = normalize_to_uint8(recon, perc_min=0.1, perc_max=99.9)

    # Save
    img_out = np.hstack((img2, img3)) 
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