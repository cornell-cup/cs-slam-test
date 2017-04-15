import numpy as np
import cv2
import time

def loop():
    vc1 = cv2.VideoCapture(2)
    vc1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vc1.set(cv2.CAP_PROP_FPS, 30)

    #cv2.namedWindow("camera1")

    vc2 = cv2.VideoCapture(1)
    vc2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vc2.set(cv2.CAP_PROP_FPS, 30)

    #cv2.namedWindow("camera2")

    fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    fourcc2 = cv2.VideoWriter_fourcc(*'XVID')

    video_write1 = cv2.VideoWriter("left.avi", fourcc1, 10, (1280, 720))
    video_write2 = cv2.VideoWriter("right.avi", fourcc2, 10, (1280, 720))

    cur_time = time.time()
    
    while True:
        
        ret1, img1 = vc1.read();
    
        ret2, img2 = vc2.read();
        
        key = cv2.waitKey(1)
        if key == 113:
            break

        #if img1 is not None:
            #cv2.imshow("camera1", img1)
            #video_write1.write(img1)

        #if img2 is not None:
            #cv2.imshow("camera2", img2)
            #video_write2.write(img2)
        
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

    
