import cv2
import numpy as np
from numba import cuda


@cuda.jit
def connected_component(image: np.array, labels: np.array) -> None:
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:
        if image[x, y] == 0:
            labels[x, y] = 0
        else:
            labels[x, y] = x * image.shape[1] + y


@cuda.jit
def update_labels(labels: np.array) -> np.array:
    x, y = cuda.grid(2)
    if x < labels.shape[0] and y < labels.shape[1]:
        label = labels[x, y]
        while label != labels[label // labels.shape[1], label % labels.shape[1]]:
            label = labels[label // labels.shape[1], label % labels.shape[1]]
        labels[x, y] = label


def find_connected_components(image: np.ndarray) -> np.array:
    # Apply Otsu's thresholding
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    labels = np.zeros_like(image, dtype=np.uint32)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    d_img = cuda.to_device(image)
    d_labels = cuda.to_device(labels)
    connected_component[blocks_per_grid, threads_per_block](d_img, d_labels)
    update_labels[blocks_per_grid, threads_per_block](d_labels)
    d_labels.copy_to_host(labels)

    return labels


def draw_output_image(labels: np.array):
    # Create the output image with color channels
    output_img = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    colors = [(0, 0, 0)] + [(255, 255, 255)] * np.max(labels)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            label = labels[i, j]
            if label != 0:
                output_img[i, j] = colors[label]

    cv2.imshow("Connected Components", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load the image
img_path = "./images/img.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Call the function for connected components
labels = find_connected_components(img)
# Draw the connected components image
draw_output_image(labels)