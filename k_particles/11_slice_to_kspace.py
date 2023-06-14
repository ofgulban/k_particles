"""Take our one slice."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/Documents/temp-k_particles/nii_prep/slice-235.nii.gz"

OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii_prep"
OUTPATH = os.path.join(OUTDIR, "step-01.png")

# =============================================================================

def normalize_to_uint8(array, perc_min=0, perc_max=100):
    val_min, val_max = np.percentile(array, [perc_min, perc_max])
    array[array > val_max] = val_max
    array = (array - val_min) / (val_max - val_min) * 255
    return array.astype(np.uint8)

# =============================================================================
create_output_directory(OUTDIR)

# Load data
nii1 = nb.load(FILE1)
data = np.asarray(nii1.dataobj, dtype=np.float32).T[::-1, :]
img1 = normalize_to_uint8(data)


# Go to k space
kspace = np.zeros(data.shape, dtype=complex)
temp = sp.fft.fft2(data)
temp = sp.fft.fftshift(temp)
kspace[:, :] = temp
img2 = normalize_to_uint8(np.imag(np.log10(kspace)))

print(np.percentile(img1, (0, 100)))
print(np.percentile(img2, (0, 100)))
save_frame_as_png(img1, img2, OUTDIR, 1, zoom_factor=2)

print("Finished.")