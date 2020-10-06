import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class ImageBGR:

    def __init__(self, file: str = None, image: np.ndarray = None):
        if isinstance(file, str):
            self.__image = cv.imread(file)  # načtení obrazu ze souboru - přepsat
        elif isinstance(image, np.ndarray):
            self.__image = image  # načtení obrazu z předaného np.ndarray - přepsat
        else:
            raise AttributeError("No valid input parameters")  # Dopsat chybu

    @classmethod
    def from_file(cls, file):
        return cls(file=file)

    @classmethod
    def from_array(cls, img_array):
        return cls(image=img_array)

    def gray(self) -> np.ndarray:
        """
        Funkce která vrací obraz ve stupních šedi    
        """
        return cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)

    def lab(self) -> np.ndarray:
        """
        Funkce která vrací obraz v barevném prostoru Lab    
        """
        return cv.cvtColor(self.__image, cv.COLOR_BGR2LAB)

    def rgb(self) -> np.ndarray:
        """
        Funkce která vrací obraz v RGB   
        """
        return cv.cvtColor(self.__image, cv.COLOR_BGR2RGB)

    def bgr(self) -> np.ndarray:
        """
        Funkce která vrací obraz v BGR
        """
        return self.__image

    def resize(self, width: int, height: int) -> 'ImageBGR':
        """
        Funkce která vrací novou instanci ImageBGR obsahující obraz z původní instance třídy ImageBGR ale s novými rozměry width a height.
        """
        new_img = cv.resize(src=self.__image, dsize=(width, height), interpolation=cv.INTER_AREA)
        return ImageBGR.from_array(new_img)

    def rotate(self, angle: int, keep_ratio: bool) -> 'ImageBGR':
        """
        Funkce která vrací novou instanci ImageBGR obsahující obraz z původní instance třídy ImageBGR ale s novými rozměry width a height.
        Pokud je nastaveno keep_ratio na True, nový obraz musí mít stejný rozměr jako původní. Pokud je nastaveno na False, nový obraz musí
        obsahovat celou obrazovou informaci z původního obrazu.
        """
        height, width, _ = self.__image.shape
        center = (width//2, height//2)
        transform_matrix = cv.getRotationMatrix2D(center, angle, 1)
        if keep_ratio:
            new_img = cv.warpAffine(self.__image, transform_matrix, (width, height))
        else:
            cos = abs(np.cos(np.deg2rad(angle)))
            sin = abs(np.sin(np.deg2rad(angle)))
            new_height = int(height*cos + width*sin)
            new_width = int(height*sin + width*cos)
            new_center = (new_width//2, new_height//2)
            transform_matrix[:, 2] += np.array(new_center) - np.array(center)
            new_img = cv.warpAffine(self.__image, transform_matrix, (new_width, new_height))
        return ImageBGR.from_array(new_img)

    def histogram(self) -> np.ndarray:
        """
        Funkce vrací histogram obrazu z jeho verze ve stupních šedi.
        """
        img_grayscale = self.gray()
        hist = cv.calcHist([img_grayscale], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
        return hist

    def perspective_transform(self, src_points, dst_points):
        M = cv.getPerspectiveTransform(src_points, dst_points)
        size = tuple(dst_points[3])  #warped image size is given by the bottom right point
        warped_img = cv.warpPerspective(self.__image, M, dsize=size)
        return warped_img

    @property
    def shape(self) -> tuple:
        """
        Funkce dekorovaná jako atribut která vrací rozměry uloženého obrazu.   
        """

        return tuple([*self.__image.shape])  # přepsat

    @property
    def size(self) -> int:
        """
        Funkce dekorovaná jako atribut která vrací obrazem obsazenou paměť (čistě polem do kterého je obraz uložen).   
        """
        return self.__image.itemsize * self.__image.size

#%%
def plot_channels(img):
    fig, axs = plt.subplots(nrows=2, ncols=2, tight_layout=True, sharex=True, sharey=True)
    p0 = axs[0, 0].imshow(img[:, :, 0], cmap='cividis', vmin=0, vmax=255)
    axs[0, 0].set_title('Channel 1')
    plt.colorbar(p0, ax=axs[0, 0])
    p1 = axs[0, 1].imshow(img[:, :, 1], cmap='cividis', vmin=0, vmax=255)
    axs[0, 1].set_title('Channel 2')
    plt.colorbar(p1, ax=axs[0, 1])
    p2 = axs[1, 0].imshow(img[:, :, 2], cmap='cividis', vmin=0, vmax=255)
    plt.colorbar(p2, ax=axs[1, 0])
    axs[1, 0].set_title('Channel 3')
    p3 = axs[1, 1].imshow(img[:, :, :])
    axs[1, 1].set_title('Combined (as RGB)')
    plt.show()

#%%
img = ImageBGR(file='img_monitors.jpg')
plt.imshow(img.rgb())
plt.show()

img_rot = img.rotate(-22, keep_ratio=False)
img_lab = img_rot.lab()
print(img_lab.shape)
plot_channels(img_lab)

histogram = img.histogram()
fig, axs = plt.subplots(2, 1)
axs[0].fill_between(range(256), np.ravel(np.zeros([256, 1])), np.ravel(histogram))
histogram_plt = axs[1].hist(np.ravel(img.gray()), bins=256)
plt.show()

# Corner order -> top left , top right, bottom left, bottom right
source_vertices = np.array([[722,800], [1632, 397], [989, 1323], [1863, 866]], dtype='float32')
d_size = (500, 250)
destination_vertices = np.array([[0, 0], [d_size[0], 0], [0, d_size[1]], [d_size[0], d_size[1]]], dtype='float32')
img_warp = img.perspective_transform(source_vertices, destination_vertices)
plt.figure()
plt.imshow(img_warp)
plt.show()
