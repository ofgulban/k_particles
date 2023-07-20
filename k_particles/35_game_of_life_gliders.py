"""Glider guns but does not really work."""

import subprocess
import numpy as np
from core import *

DIMS = [128, 128]
OUTDIR = "/Users/faruk/Documents/temp-k_particles/35_game_of_life_glider"

NR_FRAMES = 600
FRAMERATE = 30

LIVE_RATIO = 1/10

# =============================================================================
# Initialize world
world_live = create_glider_gun(DIMS[0])
img1 = world_live * 255
save_frame_as_png_1(img1, OUTDIR, "00_initialization")

for f in range(NR_FRAMES):
    # Dead cells with live neighbors
    # Compute 1-jump neighbors
    temp1 = np.roll(world_live, 1, axis=0) 
    temp2 = np.roll(world_live, 1, axis=1)
    temp3 = np.roll(world_live, -1, axis=0) 
    temp4 = np.roll(world_live, -1, axis=1)

    # Compute 2-jump neighbors
    temp5 = np.roll(world_live, 1, axis=0)
    temp5 = np.roll(temp5, 1, axis=1)
    temp6 = np.roll(world_live, -1, axis=0)
    temp6 = np.roll(temp6, -1, axis=1)
    temp7 = np.roll(world_live, 1, axis=0)
    temp7 = np.roll(temp7, -1, axis=1)
    temp8 = np.roll(world_live, -1, axis=0)
    temp8 = np.roll(temp8, 1, axis=1)

    temp = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8

    world_dead = np.zeros(DIMS, dtype=int)
    world_dead = world_live + temp
    world_dead[temp != 0] = 1
    world_dead -= world_live

    # Compute relevant cells
    world = world_dead + world_live
    idx = np.asarray(np.where(world))

    # Compute neighbors
    world_new = np.zeros(DIMS, dtype=int)
    for i in range(idx.shape[1]):
        x, y = idx[0, i], idx[1, i]

        # Compute 1-jump neighbors
        n1 = world_live[(x+1)%DIMS[0], y]
        n2 = world_live[(x-1)%DIMS[0], y]
        n3 = world_live[x, (y+1)%DIMS[1]]
        n4 = world_live[x, (y-1)%DIMS[1]]

        # Compute 2-jump neighbors
        n5 = world_live[(x+1)%DIMS[0], (y+1)%DIMS[0]]
        n6 = world_live[(x+1)%DIMS[0], (y-1)%DIMS[0]]
        n7 = world_live[(x-1)%DIMS[0], (y+1)%DIMS[0]]
        n8 = world_live[(x-1)%DIMS[0], (y-1)%DIMS[0]]

        nr_neighbors = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8

        # Any live cell
        if world_live[x, y] == 1:
            # with fewer than two live neighbours dies, underpopulation.
            if nr_neighbors < 2:
                world_new[x, y] = 0
            # with two or three live neighbours lives.
            elif nr_neighbors < 4:
                world_new[x, y] = 1
            # with more than three live neighbours dies, overpopulation.
            elif world_live[x, y] == 1:
                world_new[x, y] = 0
        # Any dead cell with exactly three live neighbours becomes a live cell
        elif nr_neighbors == 3:
            world_new[x, y] = 1

    # Convert to image format
    img2 = world_new * 255
    save_frame_as_png_1(img2, OUTDIR, i=f)

    # Update the word of the living
    world_live = np.copy(world_new)

# # =============================================================================
command = "ffmpeg -y "
command += "-framerate {} ".format(int(FRAMERATE))
command += "-i {}/frame-%04d.png ".format(OUTDIR)
command += "-c:v libx264 -r 30 -pix_fmt yuv420p "
command += "{}/00_movie.mp4".format(OUTDIR)

# ffmpeg -framerate 6 -i frame-%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4

# Execute command
print("==========!!!!!!!!!!!!==========")
print(command)
print("==========!!!!!!!!!!!!==========")
subprocess.run(command, shell=True, check=True)

print("Finished.")
