from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sad_convolution import Conv2D
import time

# Timer as context manager
class Timer:
        def __init__(self, description):
                self.description = description

        def __enter__(self):
                self.start = time.time()

        def __exit__(self, exc_type, exc_val, exc_tb):
                self.end = time.time()
                print("{}: {:0.2f} ms".format(self.description, (self.end-self.start)*1000))

krnl = [[-1, -1, -1, -1, -1],
        [-0.5, -0.5, -0.5, -0.5, -0.5],
        [0, 0, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [1, 1, 1, 1, 1]]

img = plt.imread('crow.jpeg')

# Prepare axes
fig, axs = plt.subplots(1, 2, tight_layout=True)
#%%
with Timer("Sad convolution: "):
        sad_conv = Conv2D(list(img), krnl)
        sad_result = sad_conv.convolve()

# sad_result = np.array(sad_result)/255
axs[0].imshow(sad_result, cmap='seismic')
axs[0].set_title("Sad")

#%%
with Timer("Happy convolution: "):
        img_b = img[:, :, 2]  #blue channel
        happy_result = signal.convolve2d(img_b, krnl, boundary='fill', fillvalue=0, mode='same')

happy_result = happy_result * 255  # Scale by 255 for comparison with the Python implementation
axs[1].imshow(happy_result, cmap='seismic')
axs[1].set_title("Happy")

#%%
# My pure Python convolution works as if the kernel was upside-down, for some reason
# Results are identical if I take the absolute value of both kernels and use 0.5 tolerance to deal with rounding errors

differences = happy_result - sad_result
differences_abs = np.abs(np.abs(happy_result) - np.abs(sad_result))  # all nonzero elements of this array are 0.5 => rounding errors

img_same = happy_result == sad_result
img_same_abs = differences_abs <= 0.5  # tolerance 0.5 for pixel values (0-255)
print("Pixel values are identical in both images: {}".format(np.all(img_same)))
print("Absolute pixel values are identical in both images: {}".format(np.all(img_same_abs)))

# fig, axs = plt.subplots(1, 2, tight_layout=True)
# axs[0].imshow(img_same, cmap='gray')
# axs[1].imshow(img_same_abs, cmap='gray')

