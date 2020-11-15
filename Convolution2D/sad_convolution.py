class Conv2D:
    def __init__(self, img, kernel_array):
        # assuming square-shaped kernels with odd size (so the center kernel element is exactly defined)
        self.kernel = kernel_array
        self.kernel_size = len(kernel_array)
        self.kernel_position = [0, 0]

        img_b = [[0]*len(img[0]) for row in range(len(img))]
        # # pick the third channel (blue)
        for row_idx in range(len(img)):
            for col_idx in range(len(img[0])):
                img_b[row_idx][col_idx] = int(img[row_idx][col_idx][2]*255)  # convert to int, runs 8x slower otherwise

        self.img = img_b
        self._height = len(img)  # each element of img array is a row
        self._width = len(img[0])

        self.img_result = None

    def convolve(self):
        """
        Convolves the kernel with the image.

        Returns:
            img_result: Image result of the convolution

        """
        krnl_radius = self.kernel_size//2
        img = self.expand_img(krnl_radius)  # pad the edges with krnl_radius zeros

        # too lazy to make this an array of 0s, probably not much slower this ways anyways
        self.img_result = self.img.copy()

        # The original (0,0) point is moved to (krnl_radius, krnl_radius)
        # The for loops iterate only over pixels that correspond to the original image <=> they will never
        # index into the padding border
        for row in range(krnl_radius, self._height+krnl_radius):
            for col in range(krnl_radius, self._width+krnl_radius):

                # Iterate over valid rows, from each row pick valid columns
                img_window = [imrow[col-krnl_radius:col+krnl_radius+1] for imrow in img[row-krnl_radius:row+krnl_radius+1]]

                # The resulting image won't contain the padding border
                # Adjust the index by shifting by -krnl_radius, and assign the result of the kernel multiplication
                self.img_result[row-krnl_radius][col-krnl_radius] = self.kernel_multiply(img_window)

        return self.img_result

    def kernel_multiply(self, img_window):
        """
        Performs element-wise multiplication with the kernel on the input array

        Args:
            img_window: The img segment that should be element-wise multiplied by the kernel

        Returns:
            conv_value: The sum of element-wise products of the img segment and the kernel
        """

        conv_value = 0

        for row in range(self.kernel_size):
            for col in range(self.kernel_size):
                conv_value += self.kernel[row][col] * img_window[row][col]

        return int(round(conv_value))


    def expand_img(self, n):
        """
        Expands the image by n in each direction by appending 0s

        Args:
            img: Image as list of lists
            n: Integer defining the number of pixels that should be appended in each direction

        Returns:
            new_img: Expanded image as list of lists. Shape (orig_width+2*n, orig_height+2*n)

        """

        new_img = [[0]*(self._width+2*n) for col in range(0, self._height+2*n)]

        for row_ind, row in enumerate(new_img):
            if row_ind >= n and row_ind < self._height+n:
                new_img[row_ind][n:-n] = self.img[row_ind-n]
            else:
                pass

        return new_img
