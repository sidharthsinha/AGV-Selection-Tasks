import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('sample_output.mp4')

kernel = kernel = np.ones((5,5),np.uint8)

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #If pixel lies within the specified boundaries it turns white, otherwise black  (using cv.inRange)
    mask = cv2.inRange(hsv, (17, 0, 0), (100, 255, 255))
    erosion = cv2.erode(mask, kernel, iterations=10)
    dilation = cv2.dilate(erosion, kernel, iterations=12)
    final = cv2.erode(dilation, kernel, iterations=3)

    #Matrix containing boolean values of whether matrix is black or white (True for White and False for Black)
    imask = final > 0
    #Empty Matrix
    green = np.zeros_like(frame, np.uint8)
    #Whenever imask is True it copies the same value as frame(eventually leading to all white area turning the same as in the org pic) and whenever it is False it remains 0 ie Black
    green[imask] = frame[imask]

    cv2.imshow('win', green)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break