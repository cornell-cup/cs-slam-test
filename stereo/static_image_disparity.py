import numpy as np
import cv2

img_left = cv2.imread('good_images/ambush_5_left.jpg',1)
img_right = cv2.imread('good_images/ambush_5_right.jpg',1)

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


def compute_disp_and_show(img_left, img_right):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    disparity = sbm.compute(gray_left, gray_right)
    disparity_visual = np.zeros(disparity.shape, dtype = "uint8")
    cv2.normalize(disparity, disparity_visual, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('disp',disparity_visual)

while(True):
    # Display the resulting frame
    cv2.imshow('left',img_left)
    cv2.imshow('right',img_left)

    compute_disp_and_show(img_left, img_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
