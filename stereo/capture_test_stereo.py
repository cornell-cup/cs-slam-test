# references:
# http://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html#a3d73d5521f22b92c1d24a70c702e4e6ea21fac9fae6db6a60a940fd24cc61f081
# https://github.com/jayrambhia/Vision/blob/master/OpenCV/Python/disparity.py

import numpy as np
import cv2

# init disparity map computation
SADWindowSize = 9
p1 = 8 * 1 * SADWindowSize * SADWindowSize
p2 = 4 * p1
sbm = cv2.StereoSGBM_create(
    minDisparity = -21,
    numDisparities = 96,
    blockSize = 9,
    P1 = p1,
    P2 = p2,
    disp12MaxDiff = 1,
    preFilterCap = 63,
    uniquenessRatio = 7,
    speckleWindowSize = 50,
    speckleRange = 1
)

# init windows
cv2.namedWindow("left")
cv2.namedWindow("disp")

# init camera capture
# left
cap1 = cv2.VideoCapture(0)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
cap1.set(cv2.CAP_PROP_FPS, 30)
# right
cap2 = cv2.VideoCapture(1)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
cap2.set(cv2.CAP_PROP_FPS, 30)

def compute_disp_and_show(img_left, img_right):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    disparity = sbm.compute(gray_left, gray_right)
    disparity_visual = np.zeros(disparity.shape, dtype = "uint8")
    cv2.normalize(disparity, disparity_visual, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('disp',disparity_visual)

while(True):
    # Capture frame-by-frame camera 1
    ret, img_left = cap1.read()
    # Capture frame-by-frame camera 2
    ret, img_right = cap2.read()

    # Display the resulting frame
    cv2.imshow('left',img_left)

    compute_disp_and_show(img_left, img_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap1.release()
cap2.release()
cv2.destroyAllWindows()
