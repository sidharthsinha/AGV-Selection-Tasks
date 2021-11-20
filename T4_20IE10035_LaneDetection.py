import cv2
import numpy as np

cap = cv2.VideoCapture('sample_output.mp4')
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
kernel2 = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])

def region_of_interest(img):
    h = img.shape[0]
    w = img.shape[1]
    vertices = np.array( [ [0, h], [250, 225], [1600, 200], [w, h] ] )
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    # Fill inside the polygon
    cv2.fillPoly(mask, pts=[vertices], color=(255, 255, 255))

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


while True:
    ret, frame = cap.read()
    if frame is None:
        break

    imagroi = region_of_interest(frame)

    #Thresholded to lane or not(white or not)
    image = cv2.cvtColor(imagroi, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 0, 200])
    upper = np.array([200, 76, 255])
    mask = cv2.inRange(image, lower, upper)

    #Thresholded to background or Not
    hsv = cv2.cvtColor(imagroi, cv2.COLOR_BGR2HSV)
    # If pixel lies within the specified boundaries it turns white, otherwise black  (using cv.inRange)
    mask2 = cv2.inRange(hsv, (20, 0, 0), (50, 255, 255))

    final = cv2.bitwise_and(mask, mask2)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel=kernel)

    for i in range(0, 25):
        final = cv2.filter2D(final, -1, kernel2)

    contours, hierarchy = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(contours)):
        if(contours[i].shape[0] > 650):
            cv2.drawContours(frame, contours, i, (0,0,255), 3)

    cv2.imshow('final', frame)

    key = cv2.waitKey(1)

    if key == ord('q') or key == 27:
        break