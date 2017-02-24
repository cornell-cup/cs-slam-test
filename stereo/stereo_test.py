import numpy as np
import cv2
import time

def loop():
    vc1 = cv2.VideoCapture(0)
    vc1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    vc1.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("cameraLeft")

    vc2 = cv2.VideoCapture(1)
    vc2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    vc2.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("cameraRight")

    fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    fourcc2 = cv2.VideoWriter_fourcc(*'XVID')

    video_write1 = cv2.VideoWriter("left.avi", fourcc1, 10, (640, 360))
    video_write2 = cv2.VideoWriter("right.avi", fourcc2, 10, (640, 360))
    
    #set up stereo cam calculation 
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
    
    h = 640
    w = 360
    f = 0.8*w                          # guess for focal length
    b = w # guessing baseline
    
    cur_time = time.time()
    
    while True:
        
        ret1, img1 = vc1.read();
        ret2, img2 = vc2.read();
        
        key = cv2.waitKey(1)
        if key == 113:
            break

        if img1 is not None:
            cv2.imshow("cameraLeft", img1)
            video_write1.write(img1)

        if img2 is not None:
            cv2.imshow("cameraRight", img2)
            video_write2.write(img2)

        print(1/(time.time() - cur_time))
        cur_time = time.time()
        
        #disp = stereo.compute(img1, img2).astype(np.float32) / 16.0
        #cv2.imshow('disparity', (disp-min_disp)/num_disp)

    vc1.release()
    vc2.release()
    video_write1.release()
    video_write2.release()
    cv2.destroyAllWindows()

 
        

def main():
    print("Program starting")
    loop();

if __name__ == "__main__":
    main();

    
