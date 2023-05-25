import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

img = cv2.imread('../board.jpg', cv2.IMREAD_UNCHANGED)

plt.imshow(img)
plt.show()

sigma = 1.5


# using opencv
def gaussian_filter(image, kernel_size, sigma):
    # Create a Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)

    # Convolve the image with the Gaussian kernel
    filtered_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        filtered_image[:, :, channel] = cv2.filter2D(image[:, :, channel], -1, kernel)

    return filtered_image


# Apply the Gaussian filter
kernel_size = 5
filtered_image = gaussian_filter(image=img, kernel_size=kernel_size, sigma=sigma)  # Kernel size: 5x5, Sigma: 1.0

plt.imshow(filtered_image)
plt.show()

import scipy.ndimage

def rescale(image):
    rescaled_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        rescaled_image[:, :, channel] = rescale_intensity(image[:, :, channel], in_range=(0, 255))
        rescaled_image[:, :, channel] = (rescaled_image[:, :, channel] * 255)
    rescaled_image = rescaled_image.astype("uint8")
    return rescaled_image

def gaussian_filter_handmode(image, sigma):
    # Create a Gaussian kernel
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel_handmode(size, sigma)

    # Apply padding to the image
    pad = size // 2
    padded_image = np.zeros((image.shape[0] + pad * 2,
                             image.shape[1] + pad * 2, image.shape[2]))
    for channel in range(padded_image.shape[2]):
        padded_image[:, :, channel] = np.pad(image[:, :, channel], pad, mode='constant')

    # Convolve the padded image with the Gaussian kernel
    filtered_image = np.zeros_like(padded_image)
    for channel in range(padded_image.shape[2]):
        filtered_image[:, :, channel] = convolution(padded_image[:, :, channel], kernel)

    filtered_image = rescale(filtered_image)

    return filtered_image


def convolution(image, kernel):
    # Get dimensions of image and kernel
    height, width = image.shape
    ksize = kernel.shape[0]

    # Create an empty array for the filtered image
    filtered_image = np.zeros_like(image)

    # Perform convolution
    for i in range(height - ksize + 1):
        for j in range(width - ksize + 1):
            patch = image[i:i + ksize, j:j + ksize]
            filtered_image[i, j] = np.sum(patch * kernel)

    return filtered_image


def gaussian_kernel_handmode(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -(x ** 2 + y ** 2) / (2 * sigma ** 2)), (size, size))
    kernel /= np.sum(kernel)
    return kernel


# Apply the Gaussian filter
filtered_image = gaussian_filter_handmode(image=img, sigma=sigma)  # Sigma: 1.0

plt.imshow(filtered_image)
plt.show()