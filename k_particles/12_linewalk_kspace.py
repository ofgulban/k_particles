"""Take our one slice."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"
OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii-01_linewalk"

NR_POINTS = 128
DIMS = [128, 128]

COORD_START = [32, 0]
COORD_END = [32, DIMS[1]-1]
KAPPA = 0.

# =============================================================================
create_output_directory(OUTDIR)

# Generate walk particles
coords = np.zeros((NR_POINTS, 2))
coords[:, 0] = np.linspace(COORD_START[0], COORD_END[0], NR_POINTS, endpoint=False)
coords[:, 1] = np.linspace(COORD_START[1], COORD_END[1], NR_POINTS, endpoint=False)

# -----------------------------------------------------------------------------
# Render loop
kspace_mag = np.zeros(DIMS)
kspace_phase = np.full(DIMS, 2*np.pi)
print(np.max(kspace_phase))
for i in range(NR_POINTS):
    kspace_mag *= KAPPA

    # Instert partiles onto lattice
    kspace_mag = coordinates_to_lattice(coords[i, 0], coords[i, 1], kspace_mag, 
                                        1.0, hermitian=False)

    # Mag phase to real imaginary
    kspace = kspace_mag * np.exp(1j * kspace_phase)

    # Reconstruct image
    ispace = np.fft.ifft2(np.fft.ifftshift(kspace), norm=None)

    # Save frames
    img1 = normalize_to_uint8(np.abs(kspace))
    img2 = normalize_to_uint8(np.abs(ispace))
    save_frame_as_png(img1, img2, OUTDIR, i)

print("Finished.")