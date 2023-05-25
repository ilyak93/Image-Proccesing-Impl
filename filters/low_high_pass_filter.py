import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import float32
from skimage.exposure import rescale_intensity

img = cv2.imread('../board.jpg', cv2.IMREAD_UNCHANGED)

plt.imshow(img)
plt.show()


# using opencv
def hp_filter(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    filtered_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        filtered_image[:, :, channel] = cv2.filter2D(
            image[:, :, channel], -1, kernel, borderType=cv2.BORDER_CONSTANT)

    return filtered_image


# Apply the Gaussian filter
filtered_image1 = hp_filter(image=img)  # Kernel size: 5x5, Sigma: 1.0

plt.imshow(filtered_image1)
plt.show()

import scipy.ndimage

def rescale(image):
    rescaled_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        rescaled_image[:, :, channel] = rescale_intensity(image[:, :, channel], in_range=(0, 255))
        rescaled_image[:, :, channel] = (rescaled_image[:, :, channel] * 255)
    rescaled_image = rescaled_image.astype("uint8")
    return rescaled_image

def highpass_filter(image):
    # Create a Gaussian kernel
    input = image.astype("float32")
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    # Convolve the image with the kernel
    filtered_image = np.zeros_like(input)
    for channel in range(image.shape[2]):
        filtered_image[:, :, channel] = scipy.ndimage.convolve(
            input[:, :, channel], kernel, mode='constant', cval=0.0)
    rescaled_filtered_image = rescale(filtered_image)
    return rescaled_filtered_image


# Apply the Gaussian filter
filtered_image2 = highpass_filter(img).astype("uint8")  # Sigma: 1.0

plt.imshow(filtered_image2)
plt.show()





def convolve(image, kernel, pad):
    height, width, _ = image.shape
    ksize = kernel.shape[0]
    padded_image = np.zeros((image.shape[0] + pad * 2,
                             image.shape[1] + pad * 2, image.shape[2]),
                            dtype="float32")
    for channel in range(padded_image.shape[2]):
        padded_image[:, :, channel] = np.pad(image[:, :, channel], pad,
                                             mode='constant')
    filtered_image = np.zeros((image.shape[0] + pad * 2,
                               image.shape[1] + pad * 2, image.shape[2]),
                              dtype="float32")
    for channel in range(image.shape[2]):
        for i in range(height - ksize + 1):
            for j in range(width - ksize + 1):
                patch = image[i:i + ksize, j:j + ksize, channel]
                filtered_image[i, j, channel] = np.sum(patch * kernel)
    return filtered_image


def hp_filter_handmode(image, ksize=3):
    # Create a HP kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    # Apply padding to the image
    pad = ksize // 2

    # Convolve the padded image with the Gaussian kernel
    filtered_image = convolve(image, kernel, pad)

    rescaled_filtered_image = rescale(filtered_image)

    return rescaled_filtered_image


# Apply the Gaussian filter
filtered_image = hp_filter_handmode(image=img.astype(float))  # Sigma: 1.0

plt.imshow(filtered_image)
plt.show()

print()