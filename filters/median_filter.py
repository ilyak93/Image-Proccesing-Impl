import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../board.jpg', cv2.IMREAD_UNCHANGED)

plt.imshow(img)
plt.show()
def median_filter_handmode(image, ksize):
    # Create a Gaussian kernel

    # Apply padding to the image
    pad = ksize // 2
    padded_image = np.zeros((image.shape[0] + pad * 2,
                             image.shape[1] + pad * 2, image.shape[2]), dtype=np.uint8)
    for channel in range(padded_image.shape[2]):
        padded_image[:, :, channel] = np.pad(image[:, :, channel], pad, mode='constant')

    # Convolve the padded image with the Gaussian kernel
    filtered_image = np.zeros_like(padded_image)
    height, width, _ = image.shape
    for channel in range(padded_image.shape[2]):
        for i in range(height - ksize + 1):
            for j in range(width - ksize + 1):
                patch = image[i:i + ksize, j:j + ksize, channel]
                med = np.median(patch)
                filtered_image[i, j, channel] = med


    return filtered_image



# Apply the Gaussian filter
filtered_image = median_filter_handmode(image=img, ksize=3)  # Sigma: 1.0

plt.imshow(filtered_image)
plt.show()