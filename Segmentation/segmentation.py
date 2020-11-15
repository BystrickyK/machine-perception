import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

# %%
PATH = os.path.join("imgs")
"""
str,
folder, where are all images for homework
"""
LENGTH = 0.8
"""
float,
length (letter 'L' in diagram above), this number represents
length between plane and camera lens.
"""
RESOLUTION = (600, 397)
"""
tuple of int,
resolution of camera('w x h' in diagram above)
"""
ALPHA = np.deg2rad(15)
"""
int,
angle of camera (letter alpha in diagram above)
"""
PXL = (2 * LENGTH / RESOLUTION[0]) * np.sin(ALPHA / 2)
PXL = PXL * 1e2  # m -> cm
PXL_AREA = PXL**2  # cm2
"""
float,
side length of a pixel in meters
"""

# %% Transform the images to grayscale and create a np.array of images
dtype = np.float32

empty = cv2.imread(os.path.join(PATH, "nail_empty.jpg"))
empty_color = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
empty_color = np.array(empty_color, dtype=dtype)

error = cv2.imread(os.path.join(PATH, "nail_error.jpg"))
error_color = cv2.cvtColor(error, cv2.COLOR_BGR2GRAY)
error_color = np.array(error_color, dtype=dtype)

nails = []
for i in range(1, 6):
    im = cv2.imread(os.path.join(PATH, "nail_0{}.jpg".format(i)))
    im_color = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    nails.append(im_color)
nails = np.array(nails, dtype=dtype)

bad_imgs = np.stack((empty_color, error_color))
imgs = np.concatenate((bad_imgs, nails), axis=0)

# %%
fig1, axs1 = plt.subplots(3, 3, tight_layout=True)
axs1 = axs1.flatten()
title_str = ["Empty - no nail", "Example of error", "Example 1", "Example 2", "Example 3", "Example 4", "Example 5"]
for i, img in enumerate(imgs, start=0):
    axs1[i].set_title(title_str[i])
    axs1[i].imshow(img)


# %%
def threshold_mask(img, thresh_L, thresh_H, debug=False):
    mask_L = (img < thresh_L)
    mask_H = (img > thresh_H)
    mask = np.stack((mask_L, mask_H))
    mask = np.any(mask, axis=0)

    if debug is True:
        fig, axs = plt.subplots(2, 2, tight_layout=True)
        axs[0, 0].imshow(mask_L, vmin=0, vmax=1)
        axs[0, 1].imshow(mask_H, vmin=0, vmax=1)
        axs[1, 0].imshow(mask, vmin=0, vmax=1)
        axs[1, 1].imshow(img, vmin=0, vmax=255)

    return np.array(mask, dtype=np.float32)


def erode_img(img, kernel_size=5, iterations=1):
    """
    Erodes the image. Good for getting rid of random error pixels in the mask image.
    Args:
        img:
        kernel_size:
        iterations:

    Returns:
        img:
    """
    kernel = np.ones(kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel=kernel, iterations=iterations)
    return img


def dilate_img(img, kernel_size=5, iterations=1):
    """
    Dilutes the image.
    Args:
        img:
        kernel_size:
        iterations:

    Returns:
        img:
    """
    kernel = np.ones(kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel=kernel, iterations=iterations)
    return img


def close_img(img, kernel_size=5, iterations=1):
    """
   Closing an image means dilating then eroding the image. This sequence of operations "closes" the holes in the mask.

   Args:
       img: image
       kernel_size (int): Length of one side of the kernel. Kernel will be square-shaped
       iterations (int): The number of times dilation and erosion operations will be applied.
         For example with iterations=2 -> (dilate -> dilate -> erode -> erode).

   Returns:
        img: image with closed holes
   """
    kernel = np.ones(kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel, iterations=iterations)
    return img


# %% Pick out the objects in the image (segmentation)
thresh_L = 180
thresh_H = 230

kernel_size = 5
iterations = 7

masks = []
fig2, axs2 = plt.subplots(3, 3, tight_layout=True, figsize=(16, 16))
axs2 = axs2.flatten()
for i, img in enumerate(imgs, start=0):
    axs2[i].set_title(title_str[i])
    img = threshold_mask(img, thresh_L, thresh_H)  # Brightness-based thresholding
    img = close_img(img, kernel_size, iterations)  # Closes holes in the mask
    img = erode_img(img)  # Removes single error pixels
    axs2[i].imshow(img, vmin=0, vmax=1)
    masks.append(img)
masks = np.array(masks, dtype=np.float32)

#%% Label objects in the image

def find_objects(img):
    img = np.uint8(img)
    num_objects, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(img)  # The function has two outputs
    # Background is assumed to be the first object -> trash it before returning
    return num_objects-1, labeled_img, stats[1:, :], centroids[1:, :]

def annotate_objects(ax, stats):
    for i, stat in enumerate(stats):
        [leftmost, topmost, width, height, area] = stat
        area_cm2 = area * PXL_AREA
        length = np.sqrt(width**2 + height**2) * PXL
        rect = patches.Rectangle((leftmost, topmost), width, height,
                                 linewidth=1.5, edgecolor='r', facecolor='none')
        annotation_str = "Object {}\nArea {:0.1f}cm2\nLength {:0.1f}cm".format(i+1, area_cm2, length)
        ax.text(leftmost, topmost, annotation_str, bbox=dict(facecolor='white', alpha=0.3))
        ax.add_patch(rect)


#%% add two images with more than one nail (by combining images)
new_img = np.any(np.stack([masks[4], masks[5]]), axis=0)  # Combine 5th and 6th boolean image
new_img = new_img[np.newaxis, :]
masks = np.concatenate([masks, new_img], axis=0)
new_img = np.any(np.stack([masks[2], masks[3]]), axis=0)
new_img = new_img[np.newaxis, :]
masks = np.concatenate([masks, new_img], axis=0)

#%% Find separate objects in the image and annotate them
fig3, axs3 = plt.subplots(3, 3, tight_layout=True, figsize=(16, 16))
axs3 = axs3.flatten()
for i, img in enumerate(masks, start=0):
    num_objects, img, stats, centroids = find_objects(img)
    axs3[i].set_title(str(num_objects))
    axs3[i].imshow(img, vmin=0, vmax=num_objects)
    annotate_objects(axs3[i], stats)

