import os
import numpy as np
from scipy.ndimage import zoom
import cv2
import subprocess
import random


def create_output_directory(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print("\n  Output directory:\n    {}\n".format(outdir))


def save_frame_as_png_1(img1, outdir, tag="frame", i=None, zoom_factor=8):
    img1 = zoom(img1, zoom_factor, order=0)
    if i is None:
        outname = os.path.join(outdir, "{}.png".format(tag))
    else:
        outname = os.path.join(outdir, "{}-{}.png".format(tag, str(i).zfill(4)))
    cv2.imwrite(outname, img1)

def save_frame_as_png_2(img1, img2, outdir, i, zoom_factor=8):
    img_out = np.hstack((img1, img2))
    img_out = zoom(img_out, zoom_factor, order=0)
    outname = os.path.join(outdir, "frame-{}.png".format(str(i).zfill(4)))
    cv2.imwrite(outname, img_out)


def generate_points_on_circle(center, radius, start, end, num_points):
    coords = np.zeros((num_points, 2))
    theta = np.linspace(start, end, num_points, endpoint=False)
    coords[:, 0] = center[0] + radius * np.cos(theta)
    coords[:, 1] = center[1] + radius * np.sin(theta)
    return coords


def gaussian_interpolation(img, x, y, alpha=1.0, hermitian=True):
    # Find the four neighboring grid points
    i = int(np.round(x))
    j = int(np.round(y))

    # Compute distances to each pixel (NOTE: Part of operations is redundant)
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
    img[i-1, j-1] = w00 * alpha
    img[i-1, j] = w01 * alpha
    img[i-1, j+1] = w02 * alpha
    img[i, j-1] = w10 * alpha
    img[i, j] = w11 * alpha
    img[i, j+1] = w12 * alpha
    img[i+1, j-1] = w20 * alpha
    img[i+1, j] = w21 * alpha
    img[i+1, j+1] = w22 * alpha

    # Hermitian symmetry
    if hermitian:
        img += img[::-1, ::-1]

    # img += img[::-1, :]
    # img += img[:, ::-1]

    return img


def coordinates_to_lattice(coord_x, coord_y, lattice, alpha, hermitian=True):
    # Insert points onto a clean lattice
    temp = np.zeros(lattice.shape)
    lattice += gaussian_interpolation(temp, coord_x, coord_y, alpha, hermitian)

    # Clip
    lattice[lattice > 1] = 1
    return lattice


def normalize_to_uint8(array, perc_min=0, perc_max=100):
    temp = array[array != 0]
    val_min, val_max = np.percentile(temp, [perc_min, perc_max])
    array[array > val_max] = val_max
    array[array < val_min] = val_min    
    array = (array - val_min) / (val_max - val_min) * 255
    return array.astype(np.uint8)


def Lp_norm(x1, x2, p):
    return np.power(np.abs(np.power(x1, p)) + np.abs(np.power(x2, p)), 1./p)


def create_2D_array_coordinates(size_x, size_y, center_x, center_y):
    x = np.arange(center_x - size_x // 2, center_x + size_x // 2)
    y = np.arange(center_y - size_y // 2, center_y + size_y // 2)
    return np.transpose(np.asarray(np.meshgrid(x, y)), (1, 2, 0))


def make_movie(outdir, tag="00_movie", framerate=30):
    command = "ffmpeg "
    command += "-y -framerate {} ".format(framerate)
    command += "-i {}/frame-%04d.png ".format(outdir)
    command += "-c:v libx264 -pix_fmt yuv420p "
    command += "{}/{}.mp4".format(outdir, tag)

    # Execute command
    print(command)
    subprocess.run(command, shell=True, check=True)

# =============================================================================
# Hackathon additions
# =============================================================================

def generate_random_particles(dims, nr_particles, multiplier=1):
    coords = np.zeros((nr_particles, 2))
    for i in range(nr_particles):
        coords[i, 0] = dims[0]//2 + random.uniform(-1, 1) * multiplier
        coords[i, 1] = dims[1]//2 + random.uniform(-1, 1) * multiplier
    return coords


def generate_random_velocities(dims, nr_particles, normalize=True, multiplier=1):
    coords = np.zeros((nr_particles, 2))
    for i in range(nr_particles):
        coords[i, 0] = random.uniform(-1, 1) * multiplier
        coords[i, 1] = random.uniform(-1, 1) * multiplier
        
        # Normalize to unit vector
        if normalize:
            norm = np.linalg.norm(coords[i, :])
            coords[i, :] /= norm

    return coords

def update_particle_coordinates(coords, velo, dims):
    coords += velo

    # x and y as lon as they are the same    
    coords[coords < 0] += dims[0]-2
    coords[coords >= dims[0]-2] -= dims[0]-2

    return coords
