"""
Functions for manipulating the grasping dataset images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize


class Image(object):
    def __init__(self, img):
        self.img = img

    def __getattr__(self, attr):
        return getattr(self.img, attr)

    def copy(self):
        return self.__class__(self.img.copy())

    def crop(self, top_left, bottom_right, resize=None):
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        if resize is not None:
            self.resize(resize)

    def cropped(self, *args, **kwargs):
        i = self.copy()
        i.crop(*args, **kwargs)
        return i

    def resize(self, shape):
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True)

    def resized(self, *args, **kwargs):
        i = self.copy()
        i.resize(*args, **kwargs)
        return i

    def rotate(self, angle, center=None):
        # ANGLE IN RADIANS
        if center is not None:
            center = (center[1], center[0])
        self.img = rotate(self.img, angle/np.pi*180, center=center, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def rotated(self, *args, **kwargs):
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i

    def show(self, ax=None, **kwargs):
        if ax:
            ax.imshow(self.img, **kwargs)
        else:
            plt.imshow(self.img, **kwargs)
            plt.show()

    def zoom(self, factor):
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc: self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def zoomed(self, *args, **kwargs):
        i = self.copy()
        i.zoom(*args, **kwargs)
        return i


class DepthImage(Image):
    def __init__(self, img):
        super(DepthImage, self).__init__(img)

    @classmethod
    def from_pcd(cls, pcd_filename, shape, default_filler=0, index=None):
        """
            Create a depth image from the unstructured PCD.
            If index isn't specified, use euclidean distance, otherwise choose x/y/z=0/1/2
        """
        img = np.zeros(shape)
        if default_filler != 0:
            img += default_filler

        with open(pcd_filename) as f:
            for l in f.readlines():
                ls = l.split()

                if len(ls) != 5:
                    # Not a point line in the file.
                    continue
                try:
                    # Not a number, carry on.
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])
                r = i // shape[1]
                c = i % shape[1]

                if index is None:
                    x = float(ls[0])
                    y = float(ls[1])
                    z = float(ls[2])

                    img[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                else:
                    img[r, c] = float(ls[index])

        return cls(img)

    def inpaint(self, missing_value=0):
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        self.img = cv2.copyMakeBorder(self.img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (self.img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(self.img).max()
        self.img = self.img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        self.img = cv2.inpaint(self.img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        self.img = self.img[1:-1, 1:-1]
        self.img = self.img * scale

    def gradients(self):
        # depth = cv2.GaussianBlur(depth, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_DEFAULT)
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

        return DepthImage(grad_x), DepthImage(grad_y), DepthImage(grad)

    def normalise(self):
        self.img = np.clip((self.img - self.img.mean()) / 1000.0, -1, 1)


class WidthImage(Image):
    def zoom(self, factor):
        super(WidthImage, self).zoom(factor)
        self.img = self.img/factor
