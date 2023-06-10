import subprocess
import os

SOURCE = [
    # "/Users/faruk/Documents/temp-k_particles/07a_random_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/07b_random_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/07c_random_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/07d_random_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/07e_random_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/07f_random_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/07g_random_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/07h_random_trajectory",
    # "/Users/faruk/Documents/temp-k_particles/07i_random_trajectory",
    "/Users/faruk/Documents/temp-k_particles/07j_random_trajectory",
    "/Users/faruk/Documents/temp-k_particles/07k_random_trajectory",
    "/Users/faruk/Documents/temp-k_particles/07l_random_trajectory",
    "/Users/faruk/Documents/temp-k_particles/07m_random_trajectory",
    "/Users/faruk/Documents/temp-k_particles/07n_random_trajectory",
    ]   

OUTNAMES = [
    # "07a.mp4", 
    # "07b.mp4", 
    # "07c.mp4", 
    # "07d.mp4", 
    # "07e.mp4",
    # "07f.mp4",
    # "07g.mp4",
    # "07h.mp4",
    # "07i.mp4",
    "07j.mp4",
    "07k.mp4",
    "07l.mp4",
    "07m.mp4",
    "07n.mp4",
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
    command += "-i {}/frame-%04d.png ".format(SOURCE[i])
    command += "-c:v libx264 -pix_fmt yuv420p "
    command += "{}".format(output)

    # Execute command
    print(command)
    subprocess.run(command, shell=True)

print("Finished.")
