import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

img = cv2.imread('../taj.jpg', cv2.IMREAD_UNCHANGED)

plt.imshow(img)
plt.show()

def bilateral_filter(image, sigma_spatial, sigma_intensity):
    filtered_image = np.zeros_like(image)
    height, width, channels = image.shape
    window_size = int(np.ceil(sigma_spatial)) // 2

    for y in range(height):
        for x in range(width):
            pixel_intensity = image[y, x]
            weighted_sum = np.zeros(channels, dtype=np.float32)
            normalization_factor = 0.0

            for j in range(-window_size, window_size + 1):
                for i in range(-window_size, window_size + 1):
                    neighbor_x = x + i
                    neighbor_y = y + j

                    if neighbor_x >= 0 and neighbor_x < width and neighbor_y >= 0 and neighbor_y < height:
                        neighbor_intensity = image[neighbor_y, neighbor_x]
                        spatial_distance = np.sqrt(i ** 2 + j ** 2)
                        intensity_distance = np.linalg.norm(pixel_intensity - neighbor_intensity, axis=-1)

                        weight = np.exp(-(spatial_distance ** 2) / (2 * sigma_spatial ** 2)) * \
                                 np.exp(-(intensity_distance ** 2) / (2 * sigma_intensity ** 2))

                        weighted_sum += neighbor_intensity * weight
                        normalization_factor += weight

            filtered_value = weighted_sum / normalization_factor
            filtered_image[y, x] = filtered_value

    return filtered_image


# Apply Bilateral filter
sigma_spatial = 15
sigma_intensity = 1000
#filtered_image = bilateral_filter(img, sigma_spatial, sigma_intensity)

#plt.imshow(filtered_image)
#plt.show()


def my_bilateral_filter(image, sigma_spatial, sigma_intensity):
    image = image.astype("float32")
    filtered_image = np.zeros_like(image, dtype="float32")
    height, width, channels = image.shape
    window_size = int(np.ceil(sigma_spatial))

    for y in range(window_size // 2, height - window_size // 2):
        for x in range(window_size // 2, width - window_size // 2):
            pixel_intensity = image[y, x, :]

            windows_range = list(range(- window_size // 2 + 1, window_size // 2 + 1))

            neighbors_indices_y = np.asarray(windows_range).reshape(-1, window_size).repeat(window_size, axis=0)
            neighbors_indices_x = np.asarray(windows_range).reshape(window_size, -1).repeat(window_size, axis=-1)

            neighbors_indices = np.stack((neighbors_indices_y, neighbors_indices_x), axis=-1)

            neighbor_intensities = image[y - window_size // 2 : y + window_size // 2 + 1,
                                   x - window_size // 2 : x + window_size // 2 + 1, :]

            pixel_intensity = np.zeros_like(neighbor_intensities) + pixel_intensity

            spatial_distance = np.sum((neighbors_indices ** 2), axis=-1)[..., np.newaxis].repeat(3, axis=2)

            intensity_distance = (pixel_intensity - neighbor_intensities) ** 2

            weight = np.exp(-(spatial_distance ** 2) / (2 * sigma_spatial ** 2)) * \
                     np.exp(-(intensity_distance ** 2) / (2 * sigma_intensity ** 2))

            weight[3,3, :] = (0,0,0)

            weighted_sum = np.sum((neighbor_intensities * weight).reshape(-1, 3), axis=0)

            normalization_factor = np.sum(weight.reshape(-1, 3), axis=0)

            filtered_value = weighted_sum / normalization_factor
            filtered_image[y, x] = np.round(filtered_value)

    return filtered_image

filtered_image = my_bilateral_filter(img, sigma_spatial, sigma_intensity).astype("uint8")

plt.imshow(filtered_image)
plt.show()