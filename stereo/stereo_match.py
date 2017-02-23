import numpy as np
import cv2

if __name__ == '__main__':
    print('loading images...')
    imgL = cv2.pyrDown( cv2.imread('aloeL.jpg') )  # downscale images for faster processing
    imgR = cv2.pyrDown( cv2.imread('aloeR.jpg') )

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    b = w # guessing baseline
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    dist = b * f / disp
    cv2.imshow('disp', min_disp / disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
