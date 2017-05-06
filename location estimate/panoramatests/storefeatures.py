import cv2
import numpy as np
import os

data_dir = './data' #assume in project root

detector = cv2.ORB_create()
descriptors = {}
for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        fl = file.lower()
        if fl.endswith('jpg') or fl.endswith('jpeg'):
            path = os.path.join(subdir, file)
            name = os.path.splitext(file)[0]

            img = cv2.imread(path)
            kp,des = detector.detectAndCompute(img, None)
            #imgout = cv2.drawKeypoints(img, kp, None, flags=2)
            #cv2.imwrite(os.path.join(subdir, '%skps.jpg' % (name)), imgout)

            descriptors[name] = des

print(str(descriptors))

#TODO write as JSON file to store descriptors
            
