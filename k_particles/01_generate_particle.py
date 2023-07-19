import numpy as np
import cv2

OUT1 = "/Users/faruk/Documents/temp-k_particles/01_temp_Kparticles/test1.png"
OUT2 = "/Users/faruk/Documents/temp-k_particles/01_temp_Kparticles/test2.png"
OUT3 = "/Users/faruk/Documents/temp-k_particles/01_temp_Kparticles/test3.png"

# Dimensions of the 2D wave
width = 512
height = 512

# Generate random parameters for the sine wave
frequency_x = np.random.uniform(low=0.01, high=1)
frequency_y = np.random.uniform(low=0.01, high=1)
phase_x = np.random.uniform(low=0, high=2*np.pi)
phase_y = np.random.uniform(low=0, high=2*np.pi)

# Create a grid of x and y values
x = np.linspace(0, width-1, width)
y = np.linspace(0, height-1, height)
grid_x, grid_y = np.meshgrid(x, y)

# Generate the 2D sine wave
wave = np.sin(frequency_x * grid_x + phase_x) + np.sin(frequency_y * grid_y + phase_y)
wave /= 2
print(np.percentile(wave, [0, 100]))

# Save the original wave as a PNG image using OpenCV
wave_scaled = ((wave + 1) * 127.5)
print(np.percentile(wave_scaled, [0, 100]))

# Wrap
cv2.imwrite(OUT1, wave_scaled.astype(np.uint8))

# Perform the FFT to obtain the frequency domain representation
fft_result = np.fft.fftshift(np.fft.fft2(wave))

# -----------------------------------------------------------------------------
print(np.percentile(wave, [0, 100]))
ifft_result = np.fft.fft2(np.fft.ifftshift(fft_result)).real
print(np.percentile(ifft_result, [0, 100]))
cv2.imwrite(OUT3, ifft_result)
# -----------------------------------------------------------------------------

# Calculate the magnitude spectrum
magnitude_spectrum = np.abs(fft_result)

# Scale the magnitude spectrum to 8-bit range (0-255)
scaled_spectrum = ((magnitude_spectrum / magnitude_spectrum.max()) * 255).astype(np.uint8)

# Save the magnitude spectrum as a PNG image using OpenCV
cv2.imwrite(OUT2, scaled_spectrum)

# Print a message indicating the images have been saved
print("Wave image saved as wave.png")
print("FFT image saved as fft_image.png")
