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
class NailDetector:
    def __init__(self, color_thresh=(150, 240), kernel_close=5, iterations_close=1,
                 kernel_erode=5, iterations_erode=1, kernel_dilate=5, iterations_dilate=1):
        self.thresh_L = color_thresh[0]
        self.thresh_H = color_thresh[1]

        self.kernel_close = kernel_close
        self.iterations_close = iterations_close

        self.kernel_erode = kernel_erode
        self.iterations_erode = iterations_erode

        self.kernel_dilate = kernel_dilate
        self.iterations_dilate = iterations_dilate


    def detect_nails(self, img):
        img = self.threshold_mask(img)  # Brightness-based thresholding
        img = self.close_img(img)  # Closes holes in the mask
        img = self.erode_img(img)  # Removes single error pixels
        return img

    def threshold_mask(self, img, debug=False):
        mask_L = (img < self.thresh_L)
        mask_H = (img > self.thresh_H)
        mask = np.stack((mask_L, mask_H))
        mask = np.any(mask, axis=0)

        if debug is True:
            fig, axs = plt.subplots(2, 2, tight_layout=True)
            axs[0, 0].imshow(mask_L, vmin=0, vmax=1)
            axs[0, 1].imshow(mask_H, vmin=0, vmax=1)
            axs[1, 0].imshow(mask, vmin=0, vmax=1)
            axs[1, 1].imshow(img, vmin=0, vmax=255)

        return np.array(mask, dtype='uint8')


    def erode_img(self, img):
        """
        Erodes the image. Good for getting rid of random error pixels in the mask image.
        Args:
            img:
            kernel_size:
            iterations:

        Returns:
            img:
        """
        kernel = np.ones(self.kernel_erode)
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel=kernel, iterations=self.iterations_erode)
        return np.array(img, dtype='uint8')


    def dilate_img(self, img):
        """
        Dilutes the image.
        Args:
            img:
            kernel_size:
            iterations:

        Returns:
            img:
        """
        kernel = np.ones(self.kernel_dilate)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel=kernel, iterations=self.iterations_dilate)
        return np.array(img, dtype='uint8')


    def close_img(self, img):
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
        kernel = np.ones(self.kernel_close)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel, iterations=self.iterations_close)
        return np.array(img, dtype='uint8')


# %% Pick out the objects in the image (segmentation)

detector = NailDetector(color_thresh=(170, 240), kernel_close=5, iterations_close=9)

masks = []
fig2, axs2 = plt.subplots(3, 3, tight_layout=True, figsize=(16, 16))
axs2 = axs2.flatten()
for i, img in enumerate(imgs, start=0):
    axs2[i].set_title(title_str[i])
    img = detector.detect_nails(img)
    axs2[i].imshow(img, vmin=0, vmax=1)
    masks.append(img)
masks = np.array(masks, dtype='uint8')

#%% Label objects in the image
class ObjectInfo:
    """
    Calculates moments, centroid, area, minimal area bounding rectangle, object length,
     and head/point coordinates from the object contour
    """
    def __init__(self, object_contour):
        self.object_contour = object_contour

        self.M_ = self.calculate_moments()

        self.C_ = self.calculate_centroid()

        self.A_ = self.calculate_area() * PXL_AREA

        self.min_rect = cv2.minAreaRect(self.object_contour)  # center (x,y), (width, height), angle of rotation
        self.box = cv2.boxPoints(self.min_rect)  # Corners of the rectangle
        self.length = np.max(self.min_rect[1]) * PXL  # Maximal side of the rectangle

        # Rectangle vertices that are closer to the centroid will belong to the nail's head
        vertices_dist_from_centroid = np.linalg.norm(self.box-self.C_, axis=1)
        sorted_idxs = np.argsort(vertices_dist_from_centroid)
        head_idxs = sorted_idxs[0:2]
        point_idxs = sorted_idxs[2:4]
        self.head_ = np.mean(self.box[head_idxs], axis=0)
        self.point_ = np.mean(self.box[point_idxs], axis=0)

        self.info = {'Moments': self.M_,
                     'Centroid': self.C_,
                     'Area': self.A_,
                     'Length': self.length,
                     'Head': self.head_,
                     'Point': self.point_}

    def calculate_moments(self):
        M = cv2.moments(self.object_contour)
        return M

    def calculate_centroid(self):
        Cx = int(self.M_['m10'] / self.M_['m00'])
        Cy = int(self.M_['m01'] / self.M_['m00'])
        return [Cx, Cy]

    def calculate_area(self):
        return self.M_['m00']


class NailAnnotator:
    def __init__(self):
        self.detected_objects = []

    def find_objects(self, img):
        img = np.uint8(img)

        contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if contours is not None:
            for contour in contours:
                detected_obj = ObjectInfo(contour)
                self.detected_objects.append(detected_obj)
            return True
        else:
            return False

    def annotate_objects(self, ax):
        cmap = plt.get_cmap('Pastel1', 5)
        for i, detected_obj in enumerate(self.detected_objects):

            annotation_str = "Object {}\nArea {:0.2f}cm2\nLength {:0.1f}cm".format(
                i+1, detected_obj.info['Area'], detected_obj.info['Length'])
            ax.text(detected_obj.head_[0]+20, detected_obj.head_[1]+20, annotation_str, bbox=dict(facecolor='white', alpha=0.2))

            #Bounding box
            poly = patches.Polygon(detected_obj.box, facecolor=cmap(i), edgecolor=cmap(i), alpha=0.7)
            ax.add_patch(poly)

            #Head
            head = patches.Circle(detected_obj.head_, 10, alpha=0.4, color='b')
            ax.add_patch(head)

            #Point
            point = patches.Circle(detected_obj.point_, 10, alpha=0.4, color='r')
            ax.add_patch(point)

            #Center
            center = patches.Circle(detected_obj.C_, 10, alpha=0.4, color='k')
            ax.add_patch(center)

    def forget(self):
        self.detected_objects = []

#%% add two images with more than one nail (by combining images)
new_img = np.any(np.stack([masks[4], masks[5]]), axis=0)  # Combine 5th and 6th boolean imagřee
new_img = new_img[np.newaxis, :]
masks = np.concatenate([masks, new_img], axis=0)
new_img = np.any(np.stack([masks[2], masks[3]]), axis=0)
new_img = new_img[np.newaxis, :]
masks = np.concatenate([masks, new_img], axis=0)

#%% Find separate objects in the image and annotate them
annotator = NailAnnotator()

fig3, axs3 = plt.subplots(3, 3, tight_layout=True, figsize=(16, 16))
axs3 = axs3.flatten()
for i, img in enumerate(masks, start=0):
    annotator.find_objects(img)
    axs3[i].imshow(img, vmin=0, vmax=1, cmap='Greys')
    annotator.annotate_objects(axs3[i])
    annotator.forget()

