import subprocess
import os

SOURCE = [
    # "/Users/faruk/Documents/temp-k_particles/06a_circular_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/06b_circular_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/06c_circular_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/06d_circular_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/06e_circular_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/06f_circular_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/06g_circular_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/06h_circular_trajectory",
    "/Users/faruk/Documents/temp-k_particles/06j_circular_trajectory",
    ]   

OUTNAMES = [
    # "06a.mp4", 
    # "06b.mp4", 
    # "06c.mp4", 
    # "06d.mp4", 
    # "06e.mp4", 
    # "06f.mp4", 
    # "06g.mp4",
    # "06h.mp4",
    "06j.mp4",
    ]


OUTDIR = "/Users/faruk/Documents/temp-k_particles/movies"

FRAMERATE = 30

# -----------------------------------------------------------------------------
# Output directory
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
print("  Output directory: {}\n".format(OUTDIR))

# =============================================================================

for i in range(len(SOURCE)):
    output = os.path.join(OUTDIR, OUTNAMES[i])

    command = "ffmpeg "
    command += "-y -framerate {} ".format(FRAMERATE)
    command += "-i {}/test-%04d.png ".format(SOURCE[i])
    command += "-c:v libx264 -pix_fmt yuv420p "
    command += "{}".format(output)

    # Execute command
    print(command)
    subprocess.run(command, shell=True)

print("Finished.")
