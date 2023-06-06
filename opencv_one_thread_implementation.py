import cv2

# Load the image
img_path = "images/img2.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply Otsu thresholding
_, threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Apply the Connected Components algorithm
_, labels = cv2.connectedComponents(threshold_img)

# Get the number of connected components found
num_components = labels.max()

# Draw the components in different colors
output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
colors = [(255, 255, 255)] + [(0, 0, 0)] * num_components
for i in range(1, num_components+1):
    output_img[labels == i] = colors[i]

# Display the resulting image
cv2.imshow("Connected Components", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
