import os
import numpy as np
import cv2

OUTDIR = "/Users/faruk/Documents/temp-k_particles"

DIMS = [512, 512]

PART = [250, 256]

for i in range(0, 256, 4):
	x, y = PART[0], PART[1] - i

	# Lattice
	kspace = np.zeros(DIMS, dtype=complex)

	# Insert particle
	kspace[x, y] = 1. + 0j
	kspace[512-x, 512-y] = 1. + 0j

	# Save
	img = np.real(kspace) * 255
	outname = os.path.join(OUTDIR, "test1-{}.png".format(str(i).zfill(4)))
	print(outname)
	cv2.imwrite(outname, img.astype(np.uint8))

	# Reconstruct image
	ispace = np.fft.ifft2(np.fft.ifftshift(kspace))

	# Save
	img = np.abs(ispace)
	img = img / np.max(img) * 255
	outname = os.path.join(OUTDIR, "test2-{}.png".format(str(i).zfill(4)))
	cv2.imwrite(outname, img.astype(np.uint8))

print("Finished.")
