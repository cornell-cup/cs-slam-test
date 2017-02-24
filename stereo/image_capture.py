import numpy as np
import cv2

# init windows
cv2.namedWindow("left")
cv2.namedWindow("right")

# init camera capture
# left
cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
cap1.set(cv2.CAP_PROP_FPS, 30)
# right
cap2 = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
cap2.set(cv2.CAP_PROP_FPS, 30)

picNum = 0

while(True):
    # Capture frame-by-frame camera 1
    ret, img_left = cap1.read()
    # Capture frame-by-frame camera 2
    ret, img_right = cap2.read()

    # Display the resulting frame
    cv2.imshow('left',img_left)
    cv2.imshow('right',img_left)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('images/img_left_'+picNum+'.png',img_left)
        cv2.imwrite('images/img_right_'+picNum+'.png',img_right)
        picNum+=1

# When everything done, release the capture
cap1.release()
cap2.release()
cv2.destroyAllWindows()
