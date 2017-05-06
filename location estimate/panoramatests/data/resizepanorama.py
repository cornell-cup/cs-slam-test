import cv2
import numpy as np

# resize the files to 25% of its original size

indices = np.arange(3,6)

for i in indices:
    name = 'region%d.jpg' % (i)
    img = cv2.imread(name)

    size = img.shape
    size = (int(size[1]/4), int(size[0]/4))

    imgout = cv2.resize(img, size)
    cv2.imwrite('region%d.jpg' % (i), imgout)
