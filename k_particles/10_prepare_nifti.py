"""Take our one slice."""
import os
import numpy as np
import nibabel as nb
import scipy as sp
from core import *

FILE1 = "/Users/faruk/gdrive/proj-kenshu_dataset/demo_whole_brain/sub-01_part-mag_ASPIRE_avg_stitched_composite-max.nii.gz"

SLICE_Z = 235

OUTDIR = "/Users/faruk/Documents/temp-k_particles/nii_prep"
OUTBASENAME = "slice-{}.nii.gz".format(SLICE_Z)
OUTPATH = os.path.join(OUTDIR, OUTBASENAME)

# =============================================================================
create_output_directory(OUTDIR)

# Load data
nii1 = nb.load(FILE1)
data = np.asarray(nii1.dataobj[:, :, SLICE_Z], dtype=np.float32)
print(data.shape)

# Save slice
out = nb.Nifti1Image(data, affine=np.eye(4))
nb.save(out, os.path.join(OUTDIR, OUTPATH))


print("Finished.")