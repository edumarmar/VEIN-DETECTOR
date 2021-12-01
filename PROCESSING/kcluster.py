import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion


image = cv2.imread('../veins_db/forearm/forearm3.JPG')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 2
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
# show the image

f, axarr = plt.subplots(1,2)

axarr[0].imshow(segmented_image)

# disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 1
masked_image[labels == cluster] = [255,255,255]

cluster = 0
masked_image[labels == cluster] = [21,244,238]
# convert back to original shape
masked_image = masked_image.reshape(image.shape)
#kernel = np.ones((1,1),np.uint8)
#masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel)

#masked_image = anisotropic_diffusion(masked_image, option=2, gamma=.25)
#masked_image=masked_image.astype(int)


# apply binary thresholding
masked_image=cv2.medianBlur(masked_image, 5)
ret, thresh = cv2.threshold(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY)
# visualize the binary image

cv2.imshow('Binary image', thresh)

cv2.waitKey(0)

cv2.imwrite('image_thres1.jpg', thresh)

cv2.destroyAllWindows()

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = masked_image.copy()
cv2.imshow("copy", image_copy)
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()

# show the image
#axarr[1].imshow(masked_image)
#plt.show()