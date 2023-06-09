import os
import numpy as np
import cv2
from scipy.ndimage import zoom

OUTDIR = "/Users/faruk/Documents/temp-k_particles/06g_circular_trajectory"
DIMS = [128, 128]

CENTER = (DIMS[0]-1)/2, (DIMS[1]-1)/2  # Center coordinates of the circle
RADIUS = 8  # Radius of the circle
NUM_POINTS = 1200  # Number of points to generate

# -----------------------------------------------------------------------------
# Output directory
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
print("  Output directory: {}\n".format(OUTDIR))

# =============================================================================

def generate_points_on_circle(center, radius, start, end, nr_points):
    theta = np.linspace(start, end, nr_points, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y


def gaussian_interpolation(img, x, y):
    # Find the four neighboring grid points
    i = int(np.round(x))
    j = int(np.round(y))

    p00 = abs(x - (i-1)), abs(y - (j-1))
    p01 = abs(x - (i-1)), abs(y - j)
    p02 = abs(x - (i-1)), abs(y - (j+1))

    p10 = abs(x - i), abs(y - (j-1))
    p11 = abs(x - i), abs(y - j)
    p12 = abs(x - i), abs(y - (j+1))

    p20 = abs(x - (i+1)), abs(y - (j-1))
    p21 = abs(x - (i+1)), abs(y - j)
    p22 = abs(x - (i+1)), abs(y - (j+1))

    # Calculate the weights
    w00 = np.exp(-p00[0]**2 - p00[1]**2)
    w01 = np.exp(-p01[0]**2 - p01[1]**2)
    w02 = np.exp(-p02[0]**2 - p02[1]**2)

    w10 = np.exp(-p10[0]**2 - p10[1]**2)
    w11 = np.exp(-p11[0]**2 - p11[1]**2)
    w12 = np.exp(-p12[0]**2 - p12[1]**2)

    w20 = np.exp(-p20[0]**2 - p20[1]**2)
    w21 = np.exp(-p21[0]**2 - p21[1]**2)
    w22 = np.exp(-p22[0]**2 - p22[1]**2)

    # Perform bilinear interpolation
    img[i-1, j-1] = w00
    img[i-1, j] = w01
    img[i-1, j+1] = w02
    img[i, j-1] = w10
    img[i, j] = w11
    img[i, j+1] = w12
    img[i+1, j-1] = w20
    img[i+1, j] = w21
    img[i+1, j+1] = w22

    # Hermitian symmetry
    img += img[::-1, ::-1]

    # img += img[::-1, :]
    # img += img[:, ::-1]

    return img


def coordinates_to_lattice(coords, img_dims):
    x, y = coords[0][i], coords[1][i]

    # Lattice
    kspace = np.zeros(img_dims, dtype=complex)

    # Insert particle
    kspace.real = gaussian_interpolation(kspace.real, x, y)
    return kspace


# -----------------------------------------------------------------------------
kspace = np.zeros(DIMS, dtype=complex)
for i in range(NUM_POINTS):
    kspace.real *= 0.99
    for j in range(3):
        if j % 2 == 0:
            coords = generate_points_on_circle(CENTER, RADIUS+j*16, 0, 2*np.pi, NUM_POINTS)
        else:
            coords = generate_points_on_circle(CENTER, RADIUS+j*16, 2*np.pi, 0, NUM_POINTS)
        kspace += coordinates_to_lattice(coords, DIMS)

    # Clip
    kspace.real[kspace.real > 1] = 1

    # Convert to image format
    img1 = np.abs(kspace) * 255

    # Reconstruct image
    ispace = np.fft.ifft2(kspace)

    # Convert to image format
    img2 = np.abs(ispace)
    img2 = img2 / np.max(img2) * 255
    img2 = np.fft.ifftshift(img2)  # TODO: Figure out why I need this

    # Save
    img = np.hstack((img1, img2))
    img = zoom(img, 8, order=0)
    outname = os.path.join(OUTDIR, "test-{}.png".format(str(i).zfill(4)))
    cv2.imwrite(outname, img.astype(np.uint8))

print("Finished.")
