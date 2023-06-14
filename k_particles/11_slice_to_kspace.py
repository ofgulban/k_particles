"""Take our one slice."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"

OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii_frames"
OUTPATH = os.path.join(OUTDIR, "step-01.png")

# =============================================================================

def normalize_to_uint8(array, perc_min=0, perc_max=100):
    val_min, val_max = np.percentile(array, [perc_min, perc_max])
    array[array > val_max] = val_max
    array[array < val_min] = val_min    
    array = (array - val_min) / (val_max - val_min) * 255
    return array.astype(np.uint8)


def save_3images_as_png(img1, img2, img3, outdir, i, zoom_factor=2):
    img_out = np.hstack((img1, img2, img3))
    img_out = zoom(img_out, zoom_factor, order=0)
    outname = os.path.join(outdir, "frame-{}.png".format(str(i).zfill(4)))
    cv2.imwrite(outname, img_out)


# =============================================================================
create_output_directory(OUTDIR)

# Load data
nii1 = nb.load(FILE1)
data = np.asarray(nii1.dataobj, dtype=np.float32).T[::-1, :]
img1 = normalize_to_uint8(data)


# Go to k space
kspace_base = np.zeros(data.shape, dtype=complex)
temp = sp.fft.fft2(data)
temp = sp.fft.fftshift(temp)
kspace_base[:, :] = temp
img2 = normalize_to_uint8(np.abs(kspace_base))
img3 = normalize_to_uint8(np.abs(np.angle(kspace_base)))

# -----------------------------------------------------------------------------
# K particle part
# -----------------------------------------------------------------------------
DIMS = data.shape
NR_PARTICLES = 1
KAPPA = 0
NUM_POINTS = 300
CENTER = (DIMS[0]-1)/2, (DIMS[1]-1)/2
RADIUS = 20
RHO = 10
ALPHA = 1.0

twopi = 2*np.pi
kspace = np.zeros(DIMS, dtype=complex)
for i in range(NUM_POINTS):
    kspace.real *= KAPPA
    kspace.imag *= KAPPA
    for j in range(NR_PARTICLES):
        coords = generate_points_on_circle(CENTER, RADIUS+j*RHO,
                                           twopi/4*j, twopi/4*j+twopi,
                                           NUM_POINTS)
        kspace.real = coordinates_to_lattice(coords[i, 0], coords[i, 1], 
                                             kspace.real, ALPHA)

    kspace *= kspace_base

    # Convert to image format
    img1 = normalize_to_uint8(np.abs(kspace))

    # Reconstruct image
    ispace = np.fft.ifft2(kspace)

    # Convert to image format
    img2 = normalize_to_uint8(np.abs(ispace))
    # img2 = np.fft.ifftshift(img2)  # TODO: Figure out why I need this

    save_frame_as_png(img1, img2, OUTDIR, i, zoom_factor=1)

print("Finished.")