import numpy as np
import cv2

cap = cv2.VideoCapture('Disparity.avi')

fgbg= cv2.createBackgroundSubtractor
while(cap.isOpened()):
    ret, frame = cap.read()
    fgmask=fgbg.apply(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',fgmask)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()