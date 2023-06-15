import numpy as np
import random
from core import *

DIMS = [9, 9]

OUTDIR = "/Users/faruk/Documents/temp-k_particles/08_test_gaussian"
# =============================================================================
create_output_directory(OUTDIR)

lattice = np.zeros(DIMS)
img1 = gaussian_interpolation(lattice, x=(DIMS[0]-1)/2, y=(DIMS[1]-1)/2, 
                              alpha=1.0, hermitian=False)
img1 = normalize_to_uint8(img1, perc_min=0, perc_max=100)
save_frame_as_png_1(img1, OUTDIR, i=0, zoom_factor=100)

print("Finished.")
