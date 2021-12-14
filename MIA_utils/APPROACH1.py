def despeckle(mask, cutoff):
    # find contours
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);

    # cut out small stuff
    cleaned = mask.copy();
    small_contours = [];
    for contour in contours:
        area = cv2.contourArea(contour);
        if area < cutoff:
            small_contours.append(contour);
    cv2.drawContours(cleaned, small_contours, -1, (0), -1);
    return cleaned;


def approach1(frame):
    import cv2 as cv
    lab_img = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

    # Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv.split(lab_img)

    ###########CLAHE#########################
    # Apply CLAHE to L channel
    clahe = cv.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l)

    # Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv.merge((clahe_img, a, b))

    # Convert LAB image back to color (RGB)
    CLAHE_img = cv.cvtColor(updated_lab_img2, cv.COLOR_LAB2BGR)
    CLAHE_img = cv.cvtColor(CLAHE_img, cv.COLOR_BGR2GRAY)
    filtered_img = CLAHE_img.astype("uint8")
    ret2, otsu = cv.threshold(filtered_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # despeckle black
    inverted = cv.bitwise_not(otsu);
    mask = cv.bitwise_not(despeckle(inverted, 500));

    # despeckle white
    #mask = despeckle(mask, 500);

    #opening
    #element = cv.getStructuringElement(cv.MORPH_CROSS, (11,11))
    #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, element)
    return(mask)