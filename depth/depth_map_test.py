import cv2
import numpy
from matplotlib import pyplot as plt


def get_depth(img_1, img_2):


    img_1_grey = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_grey = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img_1_grey, img_2_grey)

    return disparity.astype("uint8")

def loop():
    vc_1 = cv2.VideoCapture(0)
    vc_2 = cv2.VideoCapture(1)


    vc_1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
    vc_1.set(cv2.CAP_PROP_FPS, 30)

    vc_2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
    vc_2.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("left")
    cv2.namedWindow("right")

    while True:
        _, img_1 = vc_1.read()

        _, img_2 = vc_2.read()


        key = cv2.waitKey(1)
        if key == 27:
            break

        if img_1 is not None and img_2 is not None:
            depth_image = get_depth(img_1,img_2)
            if depth_image is not None:
                cv2.imshow("depth", depth_image)

            cv2.imshow("left", img_1)
            cv2.imshow("right", img_2)



if __name__ == "__main__":
    loop()
