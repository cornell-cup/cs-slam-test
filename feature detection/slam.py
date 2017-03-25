import cv2
import numpy

calib_left_mat = numpy.array(
    [[ 541.11082257,    0.        ,  324.08789568],
     [   0.        ,  541.11082257,  222.59189031],
     [   0.        ,    0.        ,    1.        ]])
calib_right_mat = numpy.array(
    [[ 541.11082257,    0.        ,  326.00175078],
     [   0.        ,  541.11082257,  231.72506702],
     [   0.        ,    0.        ,    1.        ]])
calib_left_dist = numpy.array(
    [[ 0.09370743,  0.22643711,  0.        ,  0.        , -0.93583831]])
calib_right_dist = numpy.array(
    [[ 0.25088046, -1.03260309,  0.        ,  0.        ,  1.41224436]])


max_features = 10

def slam(lf, rf):
    stereo = cv2.StereoSGBM_create(minDisparity = 16,
        numDisparities = 96,
        blockSize = 16,
        P1 = 8*3*3*2,
        P2 = 32*3*3*2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        mode = cv2.STEREO_SGBM_MODE_HH)
    
    luf = cv2.undistort(lf, calib_left_mat, calib_left_dist)
    ruf = cv2.undistort(lf, calib_right_mat, calib_right_dist)
    disparity = stereo.compute(luf, ruf)
        
    detector = cv2.ORB_create()
    lkp, ldesc = detector.detectAndCompute(luf, None)
    rkp, rdesc = detector.detectAndCompute(ruf, None)

    if len(lkp) > max_features:
        kpde = list(zip(lkp, ldesc))
        lkpde = sorted(kpde, key=lambda kd: kd[0].response, reverse=True)
        lkpde = lkpde[:max_features]
        lkp, ldesc = list(zip(*lkpde))
        lkp = list(lkp)
        ldesc = numpy.asarray(ldesc)

    if len(rkp) > max_features:
        kpde = list(zip(rkp, rdesc))
        rkpde = sorted(kpde, key=lambda kd: kd[0].response, reverse=True)
        rkpde = rkpde[:max_features]
        rkp, rdesc = list(zip(*rkpde))
        rkp = list(rkp)
        rdesc = numpy.asarray(rdesc)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ldesc,rdesc, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(luf,lkp,ruf,rkp,good,outImg = None,flags=2)

    return img3

if __name__=="__main__":
    lvc = cv2.VideoCapture("left.avi")
    rvc = cv2.VideoCapture("right.avi")
    
    while True:
        lret, lf = lvc.read()
        rret, rf = rvc.read()
        
        if lret and rret:
            cv2.imshow("Left", lf)
            cv2.imshow("Right", rf)
            
            res = slam(lf, rf)
            cv2.imshow("Slam", res)
    
        k = cv2.waitKey(1)
        if k == 27:
            break
