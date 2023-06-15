import numpy as np
from core import *
from config_01 import *
# from config_02 import *
# from config_03 import *

# -----------------------------------------------------------------------------
print("==========")
print("PARAMETERS")
print("==========")
print("  FRAMES    : {}".format(NUM_POINTS))
print("  CENTER    : {}".format(CENTER))
print("  RADIUS    : {}".format(RADIUS))
print("  KAPPA     : {}".format(KAPPA))
print("  ALPHA     : {}".format(ALPHA))
print("  RHO       : {}".format(RHO))
print("  NR_POINTS : {}".format(NR_PARTICLES))

# -----------------------------------------------------------------------------
create_output_directory(OUTDIR)
kspace = np.zeros(DIMS, dtype=complex)
twopi = 2*np.pi
for i in range(NUM_POINTS):
    kspace.real *= KAPPA
    for j in range(NR_PARTICLES):
        coords = generate_points_on_circle(CENTER, RADIUS+j*RHO,
                                           twopi/4*j, twopi/4*j+twopi,
                                           NUM_POINTS)
        kspace.real = coordinates_to_lattice(coords[i, 0], coords[i, 1], 
                                             kspace.real, ALPHA)

    # Convert to image format
    img1 = np.abs(kspace) * 255

    # Reconstruct image
    ispace = np.fft.ifft2(kspace)

    # Convert to image format
    img2 = np.abs(ispace)
    img2 = img2 / np.max(img2) * 255
    img2 = np.fft.ifftshift(img2)  # TODO: Figure out why I need this

    save_frame_as_png(img1, img2, OUTDIR, i)


print("Finished.")
