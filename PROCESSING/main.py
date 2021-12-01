import cv2
from matplotlib import pyplot as plt
from scipy.signal.signaltools import wiener
import numpy as np
from PIL import Image
import PIL

#LOAD IMAGE
img = cv2.imread('../veins_db/forearm/forearm3.JPG')

#APPLY CLAHE
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_img)
clahe1 = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(8, 8))
clahe1_img = clahe1.apply(l)
updated_lab_img21 = cv2.merge((clahe1_img, a, b))
lab_img = cv2.cvtColor(updated_lab_img21, cv2.COLOR_LAB2BGR)
lab_img = cv2.cvtColor(lab_img, cv2.COLOR_BGR2GRAY)

#MEDIAN FILTER
median = cv2.medianBlur(lab_img, 9)
#GAUSSIAN FILTER
gblur = cv2.GaussianBlur(median, (3, 3), 0.5)
#WEINER FILTER (DEBLURRING)
filtered_img = wiener(gblur, (1, 1))
#OTSU BINARIZATION
filtered_img=filtered_img.astype("uint8")
ret2,otsu = cv2.threshold(filtered_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#MEDIAN FILTER SECOND TIME
median2 = cv2.medianBlur(otsu, 5)

kernel = np.ones((9,9),np.uint8)
opening = cv2.morphologyEx(median2, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
erosion = cv2.dilate(closing,kernel,iterations = 1)
kernel = np.ones((9,9),np.uint8)
closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

titles = ['Original','CLAHE','Gaussian', 'Median','Wiener','OTSU','Median2','closing','erosion' ]
images = [img,lab_img,gblur, median,filtered_img, otsu,median2,closing, erosion]
save = Image.fromarray(images[5])
save.save("probarkclusters.png")
for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()