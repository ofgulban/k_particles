"""Draw spiral r = a + bθ

where:
    r is the distance from the center at a given angle θ,
    a is the starting distance from the center (offset or initial radius),
    b is the constant that controls the rate of expansion or contraction of the spiral.

"""

import numpy as np
from core import *


DIMS = 256, 256
NR_POINTS = 1000
OUTDIR = "/Users/faruk/Documents/temp-k_particles/test-spitals"

# =============================================================================
def generate_archimedean_spiral(img_dims, nr_points, a=0, b=2,
                                start=0, stop=2*np.pi):    
    # Archimedean spiral
    theta = np.linspace(start, stop * nr_points, nr_points)
    r = a + b * theta  # Compute the radius for each angle

    # Polar to carthesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Normalize to image array indices
    center_x = (img_dims[0] - 1) / 2
    center_y = (img_dims[1] - 1) / 2
    x = (x / r.max() * center_x + center_x).astype(int)
    y = (y / r.max() * center_y + center_y).astype(int)
    return x, y


x, y = generate_archimedean_spiral(DIMS, NR_POINTS, stop=10*np.pi)
img = np.zeros(DIMS)
for i in range(NR_POINTS):
    img[x[i], y[i]] = 255

# TODO: Do like 06_circular_trajectory
# TODO: Then do like nii_kspace_mask_walk

create_output_directory(OUTDIR)
save_frame_as_png_1(img, OUTDIR, tag="frame", i=None, zoom_factor=2)

