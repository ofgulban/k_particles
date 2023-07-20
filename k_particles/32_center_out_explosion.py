"""Center out explosions."""
import os
import subprocess
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

OUTDIR = "/Users/faruk/Documents/temp-k_particles/32_center_out"

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"

NR_FRAMES = 300
FRAMERATE = 30
NR_PARTICLES = 50
KAPPA = 0.75
ALPHA = 0.2

# -----------------------------------------------------------------------------
# Load data
nii1 = nb.load(FILE1)
ispace = np.asarray(nii1.dataobj, dtype=np.float32).T[::-1, :]
ispace = sp.ndimage.zoom(ispace, 0.5)
kspace = np.fft.fftshift(np.fft.fft2(ispace))
DIMS = ispace.shape
print(DIMS)

coords = generate_random_particles(DIMS, NR_PARTICLES, multiplier=10)
velo = generate_random_velocities(DIMS, NR_PARTICLES, normalize=False, 
                                  multiplier=10)

create_output_directory(OUTDIR)
mask = np.zeros(DIMS, dtype=complex)
for i in range(NR_FRAMES):
    mask *= KAPPA

    # Insert particles onto lattice
    for j in range(NR_PARTICLES):
        mask = coordinates_to_lattice(coords[j, 0], coords[j, 1], 
                                             mask, ALPHA)

    # Hermitian symmetry
    mask += mask.T
    mask /= 2

    coords = update_particle_coordinates(coords, velo, DIMS)

    if random.choice([0, 1]) == 0:
        temp = np.copy(velo)
        velo[:, 0] = temp[:, 1] * random.choice([-1, 1])
        velo[:, 1] = temp[:, 0] * random.choice([-1, 1])

    # Mask k space
    temp = kspace.real * mask + kspace.imag * mask

    # Convert to image format
    img1 = normalize_to_uint8(np.log10(np.abs(temp) + 1), perc_min=1, perc_max=99)

    # Reconstruct image
    ispace = np.fft.ifft2(temp)

    # Convert to image format
    img2 = np.abs(ispace)
    img2 = img2 / np.max(img2) * 255

    save_frame_as_png_2(img1, img2, OUTDIR, i)

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