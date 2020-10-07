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
        center = (width // 2, height // 2)
        transform_matrix = cv.getRotationMatrix2D(center, angle, 1)
        if keep_ratio:
            new_img = cv.warpAffine(self.__image, transform_matrix, (width, height))
        else:
            cos = abs(np.cos(np.deg2rad(angle)))
            sin = abs(np.sin(np.deg2rad(angle)))
            new_height = int(height * cos + width * sin)
            new_width = int(height * sin + width * cos)
            new_center = (new_width // 2, new_height // 2)
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
        src_points = np.array(src_points, dtype='float32')
        dst_points = np.array(dst_points, dtype='float32')

        transform_matrix = cv.getPerspectiveTransform(src_points, dst_points)
        size = tuple(dst_points[3])  # warped image size is given by the bottom right point
        warped_img = cv.warpPerspective(self.__image, transform_matrix, dsize=size)
        return ImageBGR(image=warped_img)

    def write_image(self, filename: str):
        filename = filename+'.bmp'
        cv.imwrite(filename, self.__image)

    def show_image(self):
        plt.figure()
        plt.imshow(self.__image)
        plt.show()

    def plot_channels(self, bgr2rgb=False):
        if bgr2rgb:
            img = self.rgb()
            colors = ('Reds_r','Greens_r','Blues_r')
        else:
            img = self.__image
            colors = ('cividis', 'cividis', 'cividis')
        # Nevím jak to rychle dát do smyčky, když jsou osy 'axs' 2D array :(
        fig, axs = plt.subplots(nrows=2, ncols=2, tight_layout=True, sharex=True, sharey=True)
        p0 = axs[0, 0].imshow(img[:, :, 0], cmap=colors[0], vmin=0, vmax=255)
        axs[0, 0].set_title('Channel 1')
        plt.colorbar(p0, ax=axs[0, 0])
        p1 = axs[0, 1].imshow(img[:, :, 1], cmap=colors[1], vmin=0, vmax=255)
        axs[0, 1].set_title('Channel 2')
        plt.colorbar(p1, ax=axs[0, 1])
        p2 = axs[1, 0].imshow(img[:, :, 2], cmap=colors[2], vmin=0, vmax=255)
        plt.colorbar(p2, ax=axs[1, 0])
        axs[1, 0].set_title('Channel 3')
        p3 = axs[1, 1].imshow(img[:, :, :])
        axs[1, 1].set_title('Combined (as RGB)')
        plt.show()

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


# %%
# Show image
img = ImageBGR(file='img_monitors.jpg')

# Rotate image, transform into a different colorspace and plot channels
img = img.rotate(-22, keep_ratio=False)
img = ImageBGR.from_array(img.lab())
img.plot_channels()



# Show histograms of gray-scaled image, histogram calculation done using both the class method
# and a matplotlib .hist method for comparison
img_prague = ImageBGR('prague_day.jpg')
img_prague.plot_channels(bgr2rgb=True)
histogram = img_prague.histogram()
fig, axs = plt.subplots(2, 1)
axs[0].fill_between(range(256), np.ravel(np.zeros([256, 1])), np.ravel(histogram))
histogram_plt = axs[1].hist(np.ravel(img_prague.gray()), bins=256)
plt.show()

# Perspective transformation, source vertices picked manually from the image
# Order -> top left , top right, bottom left, bottom right
# source_vertices = np.array([[722,800], [1632, 397], [989, 1323], [1863, 866]], dtype='float32')
img_warp = ImageBGR('monitor_opencv.jpg')
source_vertices = [[2824, 317], [3488, 435], [2819, 1675], [3486, 1581]]
d_size = (500, 250)
destination_vertices = [[0, 0], [d_size[0], 0], [0, d_size[1]], [d_size[0], d_size[1]]]
img_warp = img_warp.perspective_transform(source_vertices, destination_vertices)
img_warp.show_image()

img_warp.write_image('warped_opencv')
