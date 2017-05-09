import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

detector = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

data_dir = './data'
data = []
datanames = []
datamasks = {}
for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        fl = file.lower()
        path = os.path.join(subdir, file)
        name = os.path.splitext(file)[0]
        if fl.endswith('mask.png'):
            datamasks[name] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        elif fl.endswith('jpg'):
            datanames += [name]
            data += [cv2.imread(path)]
    
datakp = []
datades = []
for name,img in zip(datanames,data):
    kp,ds = detector.detectAndCompute(img, None)
    mask = '%smask' % (name)
    if mask in datamasks:
        kp2 = []
        ds2 = []
        maskimg = datamasks[mask]
        for k,d in zip(kp,ds):
            pt = k.pt
            if maskimg[int(pt[1])][int(pt[0])][3] == 0: # get alpha channel
                kp2 += [k]
                ds2 += [d]
        print('%s has %d keypoints after masking\n' % (name, len(kp2)))
        #imgkp = cv2.drawKeypoints(img, kp2, None, flags=2)
        #cv2.imwrite('test%smaskkp.png' % (name), imgkp)
        datakp += [kp2]
        datades += [np.asarray(ds2)]
    else:
        datakp += [kp]
        datades += [ds]

test = 1
if len(sys.argv) > 1:
    test = int(sys.argv[1])
    
testind = np.arange(5,13) #1-13
#testind = np.asarray([10, 13])
testimgs = []
testkps = []
testdess = []
for i in testind:
    img = cv2.imread('test%d.jpg' % (i))
    testimgs += [img]
    kp,des = detector.detectAndCompute(img, None)
    testkps += [kp]
    testdess += [des]

for i in np.arange(0, len(testimgs)):
    dists = []
    nummatches = []
    for j in np.arange(0, len(data)):
        matches = bf.knnMatch(testdess[i], datades[j], k=2)
        good = []
        dists2 = []
        for n,m in matches:
            if n.distance < 0.75 * m.distance:
                good += [[n]]
                #dists2 += [n.distance]
            dists2 += [n.distance]
                
        dists2 = np.asarray(dists2)
        if len(dists2) == 0:
            mean = float("inf")
        else:
            mean = np.mean(dists2)

        dists += [mean]
        nummatches += [len(good)]
        #imgout = cv2.drawMatchesKnn(testimgs[i], testkps[i], data[j], datakp[j], good, None, flags=2)
        #cv2.imwrite('match%d-%d-%f.jpg' % (testind[i], j+1, mean), imgout)

        print('Test img %d distance to region %d: %f; num of matched features: %d\n' % (testind[i],j+1,mean,len(good))) 
        
    dists = np.asarray(dists)
    mind = np.argmin(dists)
    maxn = np.argmax(nummatches)

    res = mind
    if mind != maxn:
        dd = np.abs(dists[maxn] - dists[mind]) / dists[mind]
        dn = np.abs(nummatches[maxn] - nummatches[mind]) / nummatches[maxn]
        print('Test img %d inconsistent between %d and %d: dd %f, dn %f\n' % (i+1, mind+1, maxn+1, dd, dn))
        if dn >= dd:
            res = maxn
        else:
            res = mind
        
    print('Test img %d is in region %d\n' % (i+1, res+1))

    matches = bf.knnMatch(testdess[i], datades[res], k=2)
    good = []
    for n,m in matches:
        if n.distance < 0.75 * m.distance:
            good += [[n]]
    imgout = cv2.drawMatchesKnn(testimgs[i], testkps[i], data[res], datakp[res], good, None, flags=2)
    cv2.imwrite('match%d-%d.jpg' % (testind[i], res+1), imgout)
    '''
    imgplot = cv2.imshow('matches', imgout)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
