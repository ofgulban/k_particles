import numpy as np
import cv2

OUT1 = "/home/faruk/data2/test-k_particles/img1.png"
OUT2 = "/home/faruk/data2/test-k_particles/img2.png"

# Dimensions of the 2D image
width = 512
height = 512

# Generate a random 2D image
random_image = np.random.randint(low=0, high=256, size=(height, width), dtype=np.uint8)

# Save the random image as a PNG file using OpenCV
cv2.imwrite(OUT1, random_image)

# Perform the FFT to obtain the frequency domain representation
fft_result = np.fft.fftshift(np.fft.fft2(random_image))

# Perform the inverse FFT to reconstruct the original image
reconstructed_image = np.fft.ifft2(np.fft.ifftshift(fft_result)).real

# Scale the reconstructed image to 8-bit range (0-255)
scaled_image = reconstructed_image.astype(np.uint8)

# Save the reconstructed image as a PNG file using OpenCV
cv2.imwrite(OUT2, scaled_image)

# Print a message indicating the images have been saved
print("Random image saved as random_image.png")
print("Reconstructed image saved as reconstructed_image.png")
