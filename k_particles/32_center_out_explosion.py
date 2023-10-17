"""Center out explosions."""
import os
import subprocess
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

OUTDIR = "/Users/faruk/Documents/temp-k_particles/32_center_out"

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"

NR_FRAMES = 600
FRAMERATE = 30
NR_PARTICLES = 10
KAPPA = 0.9
ALPHA = 0.2

PARTICLE_SPREAD = 0
VELOCITY_MULTIPLIER = 2

HERMITIAN = True
CHANGE_DIRECTIONS = False

# -----------------------------------------------------------------------------
# Load data
nii1 = nb.load(FILE1)
ispace = np.asarray(nii1.dataobj, dtype=np.float32).T[::-1, :]
ispace = sp.ndimage.zoom(ispace, 0.5)
kspace = np.fft.fftshift(np.fft.fft2(ispace))
DIMS = ispace.shape
print(DIMS)

coords = generate_random_particles(DIMS, NR_PARTICLES, 
                                   multiplier=PARTICLE_SPREAD)
velo = generate_random_velocities(DIMS, NR_PARTICLES, normalize=False, 
                                  multiplier=VELOCITY_MULTIPLIER)

create_output_directory(OUTDIR)
mask = np.zeros(DIMS, dtype=complex)
for i in range(NR_FRAMES):
    mask *= KAPPA

    # Insert particles onto lattice
    for j in range(NR_PARTICLES):
        mask = coordinates_to_lattice(coords[j, 0], coords[j, 1], 
                                             mask, ALPHA)

    # if HERMITIAN: # Hermitian symmetry
    #     mask = mask + mask[::-1, ::-1]
    #     mask /= 2

    if HERMITIAN: # Hermitian symmetry
        mask = mask + mask[::-1, ::-1] + mask[:, ::-1] + mask[::-1, :]
        mask /= 4

    coords = update_particle_coordinates(coords, velo, DIMS)

    if CHANGE_DIRECTIONS:
        if np.random.choice([0, 1], p=[0.99, 0.01]) == 0:
            temp = np.copy(velo)
            velo[:, 0] = temp[:, 1] * random.choice([-1, 1])
            velo[:, 1] = temp[:, 0] * random.choice([-1, 1])

    # -------------------------------------------------------------------------
    kspace_temp = np.copy(kspace)

    # Real imaginary to magnitude and phase
    kspace_mag = np.abs(kspace_temp)
    kspace_pha = np.angle(kspace_temp)

    # Mask magnitude
    kspace_mag = kspace_mag * mask

    # Calculate real and imaginary parts
    kspace_real = kspace_mag * np.cos(kspace_pha)
    kspace_imag = kspace_mag * np.sin(kspace_pha)

    # Create complex numbers using real and imaginary parts
    kspace_temp = kspace_real + 1j * kspace_imag
    recon = np.fft.ifft2(np.fft.ifftshift(kspace_temp))

    # -------------------------------------------------------------------------
    # Convert to image format
    img1 = normalize_to_uint8(np.log10(np.abs(kspace_temp) + 1), perc_min=1, perc_max=99)

    # Convert to image format
    img2 = np.abs(recon)
    img2 = img2 / np.max(img2) * 255

    save_frame_as_png_2(img1, img2, OUTDIR, i)

# =============================================================================
command = "ffmpeg -y "
command += "-framerate {} ".format(int(FRAMERATE))
command += "-i {}/frame-%04d.png ".format(OUTDIR)
command += "-c:v libx264 -r 30 -pix_fmt yuv420p "
command += "{}/00_movie.mp4".format(OUTDIR)

# Execute command
print(command)
subprocess.run(command, shell=True)

print("Finished.")