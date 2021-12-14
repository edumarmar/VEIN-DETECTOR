
def skeletonize(img):
    import numpy as np
    import cv2
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    return skel

def needle_tip(frame, mask):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # from gray space to hsv space
    edges = cv2.Canny(hsv, 0, 255)
    kernel = np.ones((100, 100), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    skel = skeletonize(closing)
    lines = cv2.HoughLinesP(skel, 1, np.pi / 180, 50, maxLineGap=50)
    x_0 = 10000
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < x_0:
                x_0 = x1
                y_0 = y1
        sum_grey = 0
        for i in range(-5,5):
            for j in range(-5,5):
                if (y_0+i<mask.shape[0]):
                        sum_grey += mask[y_0+i][x_0+j]
                        break
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # from gray space to hsv space
        if sum_grey/10 < 150:
            cv2.circle(mask, (x_0, y_0), 10, (0, 128, 0), -1)
        else:
            cv2.circle(mask, (x_0, y_0), 10, (0, 0, 255), -1)
        # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return mask