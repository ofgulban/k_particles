# Output directory
OUTDIR = "/Users/faruk/Documents/temp-k_particles/06a_circular_trajectory"

# K-space dimensions
DIMS = [128, 128]

# Circle parameters
CENTER = (DIMS[0]-1)/2, (DIMS[1]-1)/2
RADIUS = 8

# Number of points on the circle (euqual to number of frames)
NUM_POINTS = 300

# Trace, 0 leaves no trac, 1 leaves constant trace
KAPPA = 0.0

# Number of particles. Note this this will double due to Hermitian symmetry
NR_PARTICLES = 1

# Amplitude of the particles, maximum is 1
ALPHA = 1.0