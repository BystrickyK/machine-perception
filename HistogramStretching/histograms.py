# %%

from time import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# %%

# Timer decorator for simple profiling
def timer(fun):
    def timed_fun(*args, **kwargs):
        start_time = time()
        output = fun(*args, **kwargs)
        end_time = time()
        print("{}\t\t\t\tRuntime: {:0.1f} ms".format(fun.__name__, (end_time - start_time) * 10 ** 3))
        return output

    return timed_fun


def plot_img(img, title='image'):
    fig = plt.figure(tight_layout=True)
    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    ax1.imshow(img, cmap='gray')
    ax1.set_title(title)
    hist, _, _ = ax2.hist(img.flatten(), bins=256)
    ax3.plot(hist.cumsum() / hist.cumsum().max())


def range_by_quantiles(img, p_low, p_high, verbose=False):
    """ Function finds quantiles based on p_low and p_high

    Args:
        img (np.array): 2D matrix (numbers from 0 - 255) representing
            black&white image
        p_low (float): low quantile to map it to 0
        p_high (float): high quantile to map it to 255
        verbose (bool): Verbose output

    Returns:
        x_low (float): point where quantile defined by p_low ends
        x_high (float): point where quantile defined by p_high starts
    """
    img = img.flatten()  # transform 2D array into a long vector
    hist, bin_edges = np.histogram(img, bins=256)  # calculate histogram
    cumul_hist = hist.cumsum()
    cumul_hist = cumul_hist / cumul_hist.max()  # normalize cdf
    x_low = np.argmin(np.abs(cumul_hist - p_low))  # find low quantile
    x_high = np.argmin(np.abs(cumul_hist - p_high))  # find high quantile
    if verbose:
        fig, axs = plt.subplots(1, 2)
        axs[0].bar(range(0, 256), hist)
        axs[1].plot(cumul_hist)
    return x_low, x_high


@timer
def transform_by_lut(img, x_low, x_high, verbose=False):
    """ function transformes image based x_low and x_high

    args:
        img (np.array): 2d matrix (numbers from 0-255) representing
            black&white image
        x_low (float): point where low quantil ends
        x_high (float): point where high quantil starts
        verbose (bool): Verbose output

    Returns:
        transformed image
    """

    if verbose:
        print("x_low: {}\tx_high: {}".format(x_low, x_high))

    # Make a mapping function to transform a single pixel based on x_low and x_high values
    # the x_low and x_high are taken from the local scope of the 'transform_by_lut' function
    # on definition and will get embedded into the actual 'pixel_transformation' function
    def pixel_transformation(pixel_value):
        if pixel_value < x_low:
            new_pixel_value = 0
        elif pixel_value > x_high:
            new_pixel_value = 255
        else:
            new_pixel_value = 255 / (x_high - x_low) * (pixel_value - x_low)
        return new_pixel_value

    # map_function = np.vectorize(pixel_transformation, otypes=[int])
    # return map_function(img)

    # Use the pixel transformation function to create a lookup table
    # Use dict for quick indexing
    lookup_table = {x: pixel_transformation(x) for x in range(0, 256)}

    # # I thought lists are linked lists, but apparently they're arrays, so indexing is O(1) anyways
    # # https://wiki.python.org/moin/TimeComplexity
    # lookup_table = [pixel_transformation(x) for x in range(0, 256)]
    # lookup_table = np.array(lookup_table)  # somehow np.arrays are much slower (??)

    # Create a function that takes pixel value as input and uses it to index into lookup_table to get the result
    def pixel_lookup(pixel_value):
        return lookup_table[pixel_value]

    # Use the np.vectorize function to apply pixel_lookup to every element of the np.array
    map_function = np.vectorize(pixel_lookup, otypes=[int])
    return map_function(img)
    # I hoped the .vectorize function will make the calculations quicker, but it's apparently just a convenience
    # function that's implemented as a for loop anyways
    # https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html


# This function does computation (~run-time-calculation) for every pixel in the image, instead of lookup table indexing
@timer
def transform_by_ruc(img, x_low, x_high, verbose=False):
    """ function transformes image based x_low and x_high

    args:
        img (np.array): 2d matrix (numbers from 0-255) representing
            black&white image
        x_low (float): point where low quantil ends
        x_high (float): point where high quantil starts
        verbose (bool): Verbose output

    Returns:
        transformed image
    """

    if verbose:
        print("x_low: {}\tx_high: {}".format(x_low, x_high))

    # Make a mapping function to transform a single pixel based on x_low and x_high values
    # the x_low and x_high are taken from the local scope of the 'transform_by_lut' function
    # on definition and will get embedded into the actual 'pixel_transformation' function
    def pixel_transformation(pixel_value):
        if pixel_value < x_low:
            new_pixel_value = 0
        elif pixel_value > x_high:
            new_pixel_value = 255
        else:
            new_pixel_value = 255 / (x_high - x_low) * (pixel_value - x_low)
        return new_pixel_value

    map_function = np.vectorize(pixel_transformation, otypes=[int])
    return map_function(img)


# %%

img = plt.imread("L.jpg")
grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
P_LOW = 0
P_HIGH = 0.7
# ^^ setting the upper probability to a low enough value (around the "elbow" in the CDF)
# lets the quantile move away from the peak at 255
x_low, x_high = range_by_quantiles(grayscaled, P_LOW, P_HIGH, verbose=False)
transformed = transform_by_lut(img, x_low, x_high, verbose=False)
transformed_ruc = transform_by_ruc(grayscaled, x_low, x_high, verbose=False)
# %%

# PLOT HISTOGRAMS HERE (CHECK IF IT IS CORRECT)
plot_img(img, 'original image')
plot_img(transformed, 'transformed image | naive histogram stretching with p_high = 0.7')


# %% md
def create_mask(img, thresh_low, thresh_high, plot=False):
    """ Function creates a boolean mask with pixels whose values are between
        thresh_low and thresh_high values

    Args:
        img (np.array): 2D matrix (numbers from 0 - 255) representing
            black&white image
        thresh_low (int): lower pixel brightness threshold for boolean masking
        thresh_high (int): higher pixel brightness threshold for boolean masking
        plot (bool): verbose plotting

    Returns:
        mask_combined (np.array): 2D boolean matrix with the composite mask
    """
    mask_low, mask_high = img >= thresh_low, img <= thresh_high  # creates two masks that filter out dark/bright pixels
    masks = np.stack([mask_low, mask_high])  # stack the masks along a new axis
    mask_composite = masks.all(axis=0)  # check for pixels that fit in both masks
    if plot:
        plt.figure()
        plt.imshow(mask_composite, cmap='gray')
        plt.title("Boolean mask")
    return mask_composite


def range_by_quantiles_masked(img, p_low, p_high, thresh_low=5, thresh_high=250, verbose=False):
    """ Function finds quantiles based on p_low and p_high

    Args:
        img (np.array): 2D matrix (numbers from 0 - 255) representing
            black&white image
        p_low (float): low quantile to map it to min pixel value
        p_high (float): high quantile to map it to max pixel value
        thresh_low (int): lower pixel brightness threshold for boolean masking
        thresh_high (int): higher pixel brightness threshold for boolean masking
        verbose (bool): Verbose output
        plot (bool): verbose plotting

    Returns:
        x_low (float): point where quantil defined by p_low ends
        x_high (float): point where quantil defined by p_high starts
    """
    mask = create_mask(img, thresh_low, thresh_high)
    img = img[mask]  # pick pixels that fit the mask (flattens automatically)

    bins = 256 - thresh_low - (255 - thresh_high)
    hist, bin_edges = np.histogram(img, bins=int(bins))  # calculate histogram
    cumul_hist = hist.cumsum()
    cumul_hist = cumul_hist / cumul_hist.max()  # normalize cdf
    if verbose:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(mask, cmap='gray')
        axs[1].bar(range(thresh_low, thresh_high + 1), hist)
        axs[2].plot(range(thresh_low, thresh_high + 1), cumul_hist)
    x_low = np.argmin(np.abs(cumul_hist - p_low)) + thresh_low
    x_high = np.argmin(np.abs(cumul_hist - p_high)) + thresh_low
    return x_low, x_high, mask


@timer
def advanced_lut(img, p_low, p_high, verbose=False, plot=False, segment_borders=None, segment_count=10):
    """ Function transforms image based p_low and p_high

    Segments the image pixels based on their values and does histogram
    equalization within the segments. This lets us separate the bright
    and dark parts of the image and do histogram equalization on a separate range.

    This function also tries to do it without artifacts in transformed
    image.

    Args:
        img (np.array): 2D matrix (numbers from 0 - 255) representing
            black&white image
        p_low (float): low quantile to map it to min pixel value
        p_high (float): high quantile to map it to max pixel value
        plot (bool): verbose plotting

    Returns:
        img_new (np.array): transformed image
    """
    img_new = img.copy()  # copy the original image

    if segment_borders == None:  # segment the image into segments with equal pixel value ranges
        segment_range = 256 / segment_count
        segment_borders = [(int(left * segment_range), int(right * segment_range - 1)) for left, right in
                           enumerate(range(1, segment_count + 1))]

    for seg, thresholds in enumerate(segment_borders):
        x_low, x_high, mask = range_by_quantiles_masked(img, p_low, p_high, thresholds[0], thresholds[1],
                                                        verbose=plot)
        if verbose:
            print("Segment {}: \t x_low: {} \t x_high: {}".format(seg, x_low, x_high))
        img_masked = img[mask]  # flat array of masked pixels

        def pixel_transformation(pixel_value):
            if pixel_value < x_low:
                new_pixel_value = thresholds[0]  # pixels below the quantile are set to the lower segment threshold
            elif pixel_value > x_high:
                new_pixel_value = thresholds[1]  # ~ ^
            else:
                new_pixel_value = thresholds[0] + (thresholds[1] - thresholds[0]) / (x_high - x_low) * (
                        pixel_value - x_low)
            return new_pixel_value

        # Use the pixel transformation function to create a lookup table
        lookup_table = {x: pixel_transformation(x) for x in range(0, 256)}

        # Create a function that takes pixel value as input and uses it to index into lookup_table to get the result
        def pixel_lookup(pixel_value):
            return lookup_table[pixel_value]

        map_function = np.vectorize(pixel_lookup, otypes=[int])
        # vv update the pixels that fit the current mask, most time wasted here vv
        img_new[mask] = map_function(img_masked)

    return img_new


# Runs ~30x faster (on a single channel) because the computations don't rely on Python for loops
@timer
def masked_clahe(img, thresh_low=0, thresh_high=255):
    new_img = img.copy()
    mask = create_mask(img, thresh_low, thresh_high, plot=False)  # filter out the darkest/brightest pixels
    clahe = cv.createCLAHE()
    new_img[mask] = clahe.apply(img)[mask]  # update only masked pixels
    return new_img


# %%
# img_RGB = plt.imread("M.jpg")  # The biomedical image is more interesting for contrast stretching imo
img_RGB = plt.imread("L.jpg")
img = img_RGB.copy()
P_LOW = 0.1
P_HIGH = 0.9

# I managed to unintentionally write the functions so they work even for imgs with multiple channels
# img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # uncomment to use grayscale imgs
# transformed_img = advanced_lut(img, P_LOW, P_HIGH, verbose=True)


# stretch the histogram within the defined segments
borders = ((0, 110), (111, 255))  # values set up for the bio-medical img
transformed_img_borders = advanced_lut(img, P_LOW, P_HIGH, segment_borders=borders, verbose=False)
# If only one segment border is defined, the function works just like transform_by_lut, except on a masked image
# This way, pixels with values <5 and >240 are untouched by the transformation
transformed_img_border = advanced_lut(img, P_LOW, P_HIGH, segment_borders=((5, 240),), verbose=False, plot=True)

# Transform img into LAB, use clahe equalization on L, transform back into RGB
transformed_img_masked_clahe = cv.cvtColor(img_RGB, cv.COLOR_RGB2LAB)
transformed_img_masked_clahe[:, :, 0] = masked_clahe(
    transformed_img_masked_clahe[:, :, 0], thresh_low=3, thresh_high=240)  # local histogram equalization
transformed_img_masked_clahe = cv.cvtColor(transformed_img_masked_clahe, cv.COLOR_LAB2RGB)

# %%
plot_img(img, 'original image')
# plot_img(transformed_img, 'transformed with default arguments -> 2 equally big segments')
# plot_img(transformed_img_5, 'transformed with 5 equally big segments')
plot_img(transformed_img_borders, 'transformed with predefined segment borders')
plot_img(transformed_img_border, 'transformed with one predefined border')
plot_img(transformed_img_masked_clahe, 'transformed by clahe with default parameters')
plt.show()
