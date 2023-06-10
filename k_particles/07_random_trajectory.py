import numpy as np
import random
from core import *

OUTDIR = "/Users/faruk/Documents/temp-k_particles/07n_random_trajectory"
DIMS = [128, 128]

NR_FRAMES = 600
NR_PARTICLES = 3
KAPPA = 0.99
ALPHA = 0.2

# -----------------------------------------------------------------------------
print("==========")
print("PARAMETERS")
print("==========")
print("  FRAMES       : {}".format(NR_FRAMES))
print("  NR_PARTICLES : {}".format(NR_PARTICLES))
print("  KAPPA        : {}".format(KAPPA))
print("  ALPHA        : {}".format(ALPHA))


# -----------------------------------------------------------------------------
def generate_random_particles(dims, nr_particles):
    coords = np.zeros((nr_particles, 2))
    for i in range(nr_particles):
        # coords[i, 0] = random.uniform(1, dims[0]-2)
        # coords[i, 1] = random.uniform(1, dims[1]-2)

        coords[i, 0] = 63.5
        coords[i, 1] = 63.5
    return coords


def generate_random_velocities(dims, nr_particles):
    coords = np.zeros((nr_particles, 2))
    for i in range(nr_particles):
        coords[i, 0] = random.uniform(-1, 1)
        coords[i, 1] = random.uniform(-1, 1)
        
        # Normalize to unit vector
        norm = np.linalg.norm(coords[i, :])
        coords[i, :] /= norm

    return coords


def update_particle_coordinates(coords, velo, dims):
    coords += velo

    # x and y as lon as they are the same    
    coords[coords < 0] += dims[0]-2
    coords[coords >= dims[0]-2] -= dims[0]-2

    return coords


# -----------------------------------------------------------------------------
coords = generate_random_particles(DIMS, NR_PARTICLES)
velo = generate_random_velocities(DIMS, NR_PARTICLES)

create_output_directory(OUTDIR)
kspace = np.zeros(DIMS, dtype=complex)
twopi = 2*np.pi
for i in range(NR_FRAMES):
    kspace.real *= KAPPA

    # Insert particles onto lattice
    for j in range(NR_PARTICLES):
        kspace.real = coordinates_to_lattice(coords[j, 0], coords[j, 1], 
                                             kspace.real, ALPHA)

    coords = update_particle_coordinates(coords, velo, DIMS)

    # if i == NR_FRAMES//2:
    if random.choice([0, 1]) == 0:
    # if random.choice([0, 1, 2, 3, 4, 5]) == 0:
        temp = np.copy(velo)
        velo[:, 0] = temp[:, 1] * random.choice([-1, 1])
        velo[:, 1] = temp[:, 0] * random.choice([-1, 1])

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
