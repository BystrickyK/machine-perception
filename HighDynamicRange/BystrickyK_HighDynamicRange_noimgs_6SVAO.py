import glob

import cv2
import exifread  # I couldn't find 'exif' in conda channels
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr


# %%
# Funkce pro načtení obrázků a výpočty

def read_images(file_pattern, scale_percent=5):
    files = glob.glob(file_pattern)  # Načtení cest ke všem souborům dle daného vzoru

    # příprava listů
    imgs = []
    t = []

    # cyklus přes všechny soubory (obrázky v dané cestě)
    for file in files:
        tmp_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)  # Načtení obrázku
        width = int(tmp_img.shape[1] * scale_percent / 100)  # nový rozměr po zmenšení dle dané hodnoty v %
        height = int(tmp_img.shape[0] * scale_percent / 100)  # nový rozměr po zmenšení dle dané hodnoty v %
        dim = (width, height)  # nový rozměr po zmenšení dle dané hodnoty v %
        imgs.append(
            cv2.resize(tmp_img, dim, interpolation=cv2.INTER_AREA))  # zmenšení obrázku a uložení do listu
        with open(file, 'rb') as f:
            info = exifread.process_file(f, 'rb')  # Načtení EXIF metadat z obrázku
            try:  # The .tif images don't have exposure times in their exifs
                exposure_time = info['EXIF ExposureTime']
                exposure_time = exposure_time.values[0]  # Returns a ratio object with num, den attributes
                exposure_time = exposure_time.num / exposure_time.den  # Calculate exposure time from the ratio
                t.append(exposure_time)  # Uložení času expozice do listu
            except:
                pass
    return dim, np.array(imgs, dtype=int), np.array(
        t)  # Návrat velikosti obrázků, pole s obrázky a vektoru časů expozic


def get_weights(Z, L):
    """

    Args:
        Z: Pixels
        L: Intensity range
    Returns:

    """
    return np.interp(Z, [0, (L - 1) / 2, L - 1], [0, 1, 0])


def rgb2lab(rgb_image):
    new_image = cv2.cvtColor(rgb_image.astype('uint8'), cv2.COLOR_RGB2LAB)
    return new_image


def lab2bgr(lab_image):
    new_image = cv2.cvtColor(lab_image.astype('uint8'), cv2.COLOR_LAB2RGB)
    return new_image


def bgr2rgb(bgr_image):
    b, g, r = cv2.split(bgr_image)  # Rozdělení barevných kanálů
    return cv2.merge([r, g, b])  # Spojení v jiném pořadí a použití jako návratovou hodnotu funkce


def imgs_RGB_to_LAB(imgs):
    """
    Convert an array of images from RGB to LAB
    """
    imgs_LAB = np.array([rgb2lab(img) for img in imgs])  # Convert images to LAB
    return imgs_LAB


def sort_by_exposure_time(imgs, t):
    """
    Sorts images by exposure time
    Args:
        imgs (np.ndarray): Array of shape (P, width, height) storing images to be sorted
        t (np.ndarray): Vector of size P with exposure times for each image

    Returns:
        imgs (np.ndarray): Sorted array of images
        n (np.ndarray): Sorted vector of exposure times
    """

    sort_idx = np.argsort(t)
    imgs = imgs[sort_idx]
    t = t[sort_idx]
    return imgs, t


def calc_weights(imgs, L=256):
    """
    Calculate pixel weights based on the original pixel's intensity distance from intensity center (255/2)
    assuming range of pixel values is (0, 255). Pixels with value L/2 have weight==1, weights decrease linearly,
    reaching 0 at intensities 0 and L.

   Args:
        imgs : Array of images in the shape (P, width, height)
        L (int) : Maximum intensity value
        aggro (int) : How much should edge values be penalized

    Returns:
        imgs_w (np.ndarray) : Array of shape (P, width, height), where each "pixel" is the corresponding weight

    """
    imgs_w = []
    for img in imgs:
        img_w = get_weights(img, L)
        imgs_w.append(img_w)
    return np.array(imgs_w)


# Plot 4x4 image montage
def plot_montage(imgs, title_str='Montage', img_size=None):
    """
    Creates image montage.

    """
    if len(imgs.shape) == 3:  # If images have only 1 channel
        imgs = imgs[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, imgs, imgs), axis=3)  # Add two identical channels to each image

    if img_size == None:
        img_size = (imgs.shape[2], imgs.shape[1])  # Don't resize images

    montage = imutils.build_montages(imgs, img_size, (4, 4))
    plt.figure()
    plt.imshow(montage[0])
    plt.title(title_str)
    plt.show()


# %%
def imgs_to_Z(imgs):
    """
    Transforms an array of images to a 2D array of pixels
    Args:
       imgs (np.ndarray): Single channel image array with shape (P, width, height)

    Returns:
        (np.ndarray): Pixel intensities, Z(j,i) is i-th pixel in j-th image
    """
    Z = [img.flatten() for img in imgs]
    return np.array(Z)


def estimate_exposure(Z, weights):
    """
    https://cw.fel.cvut.cz/wiki/courses/b4m33dzo/labs/3_hdr
    Odhadněte ozáření a časy expozic z intenzit jednotlivých pixelů z více expozic.

    Předpokládejte, že odezvová funkce f je identická funkce.

    Z Intenzity pixelů, Z(j,i) je intenzita itého pixelu v jtém obrázku
    w váhy
    """

    N = Z.shape[1]  # Number of pixels in each image
    P = Z.shape[0]  # Total number of images

    # Edit weights
    weights = np.append(weights, 1)  # For the last row -> constraint
    weights = np.sqrt(weights)  # Weights will be squared in LSQR

    # Create b vector
    b = Z.flatten()
    eps = np.nextafter(0, 1)  # tiny number, size of 'minimal' step from 0 to 1
    b = np.log(b + eps)  # added eps to avoid infinities when Z==0
    b = np.append(b, 0)  # From constraint (exposure time in img 1 is 1 sec)
    b = b * weights

    # Create A matrix
    # Create index arrays
    eyes_idxs = np.array([(i, i % N) for i in np.arange(0, P * N)], dtype='uint32')  # Indexes for log(Ei)
    ones_idxs = np.array([(i, N + i // N) for i in np.arange(0, P * N)], dtype='uint32')  # Indexes for log(tj)
    constraint_idx = np.array([[P * N, N]])  # Index for the constraint equation

    nnz_elements = len(eyes_idxs) + len(ones_idxs) + len(constraint_idx)

    data = np.ones([nnz_elements, ])  # All elements will be initialized with value 1
    row = np.concatenate((eyes_idxs[:, 0], ones_idxs[:, 0], constraint_idx[:, 0]))  # Row indexes
    col = np.concatenate((eyes_idxs[:, 1], ones_idxs[:, 1], constraint_idx[:, 1]))  # Column indexes

    A = csc_matrix((data, (row, col)), shape=(N * P + 1, N + P))  # Create the sparse matrix

    # Apply weights to rows
    weights = csc_matrix(weights).transpose()  # Makes row matrix by default, transpose to get col matrix
    A = A.multiply(weights)  # Multiplies each row by the corresponding weight (element-wise multiplication)

    # Solve for x with LSQR
    sol = lsqr(A, b, show=True)

    # Extract the solution
    x = sol[0]  # Vector with the LSQR solution
    E = x[:N]  # Vector with log pixel irradiances
    t = x[N:]  # Vector with log exposure times for each image

    # _, sigma_max, _ = svds(A, 1, which='LM')  # Largest singular value
    # _, sigma_min, _ = svds(A, 1, which='SM')  # Smallest singular value
    # A_cond_r = sigma_max[0]/sigma_min[0]  # Relative condition number
    # print(f"Istop: {istop}, cond(A): {A_cond}, cond_r(A): {A_cond_r}")

    return np.exp(E), np.exp(t)


def visualize_exposure(E, t, dim):
    """
    Visualizes the results from estimate_exposure
    Args:
       E: Vector of irradiances
       t: Vector of exposure times
       dim: Shape of the image
    """

    # Plot the irradiances in log scale
    x_img = np.reshape(E, np.flip(dim))  # Vector of pixels to 2D array

    plt.figure()
    plt.imshow(np.log(x_img), cmap='jet')
    plt.colorbar()
    plt.show()

    fig, axs = plt.subplots(1, 2, tight_layout=True)
    axs[0].plot(np.sort(E))
    axs[0].set_yscale('log')
    axs[0].grid(True)
    axs[0].set_title("Irradiances (sorted)")

    # t_sort_idx = np.argsort(t)
    # axs[1].bar(t_sort_idx, t[t_sort_idx])
    axs[1].bar(range(len(t)), t)
    axs[1].set_title("Exposure times")


# %%
# Load images
dim, imgs, t = read_images('./imgs/color*.tif', 10)
imgs = imgs / (2 ** 8)  # tif channels have 16 bit encoding -> convert to 8 bit

# Show images
# plot_montage(imgs, title_str='images')

imgs_LAB = imgs_RGB_to_LAB(imgs)
imgs_L = imgs_LAB[:, :, :, 0]  # Extract L channel
plot_montage(imgs_L, title_str='LAB')

imgs_weights = calc_weights(imgs_L)
# plot_montage(imgs_weights * 255, title_str='weighted images')

weights = imgs_weights.flatten()
Z = imgs_to_Z(imgs_L)


E, t = estimate_exposure(Z, weights)
visualize_exposure(E, t, dim)

# %%
def estimate_response(Z, t, weights, lambda_=2):
    """
    Odhadněte ozáření a inverzní odezvovou funkci z intenzit jednotlivých pixelů z více expozic a časů expozic.

    Z Intenzity pixelů, Z(j,i) je intenzita itého pixelu v jtém obrázku
    t časy expozic, t > 0.
    weight váhy
    lambda_ keoficent trestu za porušení hladkosti
    """

    L = np.max(Z) + 1
    N = Z.shape[1]  # Number of pixels in each image
    P = Z.shape[0]  # Total number of images

    # Edit weights
    weights = np.sqrt(weights)  # Weights will be squared in LSQR

    t = np.log(t)  # t must be in log values for regression

    # Create A matrix and b matrix
    # Create index and data lists
    data_A = []
    data_B = []
    row = []
    col = []

    # Add system A of LHS equations from (6) -> g(Zij) - log(Ei) # add system b of RHS equations (6) -> log(tj)
    g_indexes = np.array([(i, Z.take(i)) for i in np.arange(0, N * P)])
    data_A.extend(np.ones([len(g_indexes, )]) * weights)  # apply weights immediately
    row.extend(g_indexes[:, 0])
    col.extend(g_indexes[:, 1])

    E_indexes = np.array([(i, L + i % N) for i in np.arange(0, N * P)])
    data_A.extend(-1 * np.ones([len(E_indexes, )]) * weights)  # apply weights immediately
    row.extend(E_indexes[:, 0])
    col.extend(E_indexes[:, 1])

    # add system b of RHS equations (6) -> log(tj)
    values_B = [t.take(i // N) for i in np.arange(0, N * P)]
    data_B.extend(values_B * weights)

    # add the constraint, equation (9)
    row_head = N * P
    constraint_idx = (row_head, np.floor(L / 2 + 0.5))
    row.append(constraint_idx[0])
    col.append(constraint_idx[1])
    data_A.append(1)
    data_B.append(0)


    # add fin-diff second derivative
    row_head += 1
    sqrt_lambda = np.sqrt(lambda_)  # pre-calculate np.sqrt(lambda)

    d2_indexes_subdiag = np.array([[1], [1]]) * np.arange(L - 2)
    d2_indexes_subdiag = d2_indexes_subdiag.transpose()
    d2_indexes_subdiag += np.array([row_head, 0])
    row.extend(d2_indexes_subdiag[:, 0])
    col.extend(d2_indexes_subdiag[:, 1])
    data_A.extend(1 * np.ones(L - 2) * sqrt_lambda)

    d2_indexes_diag = d2_indexes_subdiag + np.array([0, 1])  # add 1 to column indexes
    row.extend(d2_indexes_diag[:, 0])
    col.extend(d2_indexes_diag[:, 1])
    data_A.extend(-2 * np.ones(L - 2) * sqrt_lambda)

    d2_indexes_superdiag = d2_indexes_diag + np.array([0, 1])
    row.extend(d2_indexes_superdiag[:, 0])
    col.extend(d2_indexes_superdiag[:, 1])
    data_A.extend(1 * np.ones(L - 2) * sqrt_lambda)

    data_B.extend(np.zeros(L - 2))

    # Define matrices
    A = csc_matrix((data_A, (row, col)), shape=(N * P + 1 + (L - 2), L + N))  # Create the sparse A matrix
    # plt.figure()
    # A2 = A.toarray()
    # plt.imshow(A2)
    b = np.array(data_B)

    # Solve for x with LSQR
    sol = lsqr(A, b, show=True)

    # Extract the solution
    x = sol[0]  # Vector with the LSQR solution
    finv = x[:L]  # Vector with log finv
    E = x[L:]  # Vector with log irradiances for each pixel

    return np.exp(E), np.exp(finv)


# %%
# Load images
dim, imgs, t = read_images('./imgs/lampicka*.jpg', 10)
imgs, t = sort_by_exposure_time(imgs, t)
# plot_montage(imgs, title_str='images')

imgs_LAB = imgs_RGB_to_LAB(imgs)
imgs_L = imgs_LAB[:, :, :, 0]  # Extract L channel
# plot_montage(imgs_L, title_str='LAB')

imgs_weights = calc_weights(imgs_L)  # values at the intensity limits have weights 1/aggro
# plot_montage(imgs_weights * 255, title_str='weighted images')

weights = imgs_weights.flatten()
Z = imgs_to_Z(imgs_L)
E, finv = estimate_response(Z, t, weights, lambda_=1)  # Solve the system of equations

# Plot the irradiances in log scale
x_img = np.reshape(E, np.flip(dim))  # Vector of pixels to 2D array
fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].imshow(np.log(x_img), cmap='nipy_spectral')
axs[1].plot(finv)
plt.show()
