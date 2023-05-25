import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

img = cv2.imread('../dragon.jpg', cv2.IMREAD_GRAYSCALE)

plt.imshow(img)
plt.show()

sigma = 1.5



def gaussian_filter_handmode(image, sigma):
    # Create a Gaussian kernel
    size=3
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

    # Apply padding to the image
    pad = size // 2

    padded_image = np.pad(image, pad, mode='constant').astype("float32")

    # Convolve the padded image with the Gaussian kernel

    filtered_image = np.abs(convolution(padded_image, kernel)).astype("uint8")

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


from PIL import Image
im = Image.fromarray(img)
im.save("dgs.jpeg")

from PIL import Image
im = Image.fromarray(filtered_image)
im.save("dve.jpeg")