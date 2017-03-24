import cv2
import numpy

last_cap = None

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

cp_x = calib_left_mat[0][2]
cp_y = calib_left_mat[1][2]
fx = calib_left_mat[0][0]
fy = calib_left_mat[1][1]

def image_plane_to_world_space(pixel_x, pixel_y, image_center_x, image_center_y, focal_length_x, focal_length_y, depth):
    x = ((pixel_x - image_center_x)*depth) / (focal_length_x)
    y = ((pixel_y - image_center_y)*depth) / (focal_length_y)
    return [x,y,depth]


def decompose_keypoint(keypoint):
    keypoint_out = []

    keypoint_out.append(float(keypoint.pt[0]))
    keypoint_out.append(float(keypoint.pt[1]))

    return keypoint_out

def get_similar_feature_keypoints(keypoints, desc1, desc2):
    keypoints_out = []
    desc_out = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2, k = 2)

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            if m.trainIdx < len(keypoints):
                keypoints_out.append(decompose_keypoint(keypoints[m.trainIdx]))
            else:
                 keypoints_out.append(decompose_keypoint(keypoints[m.queryIdx]))
            desc_out.append([m])
    
    return (numpy.asarray(keypoints_out), desc_out)

def slam(lf, rf):
    global last_cap
    stereo = cv2.StereoSGBM_create(minDisparity = 25,
        numDisparities = 64,
        blockSize = 9,
        P1 = 8*3*3*2,
        P2 = 32*3*3*2,
        disp12MaxDiff = 1,
        preFilterCap = 63,
        uniquenessRatio = 7,
        speckleWindowSize = 50,
        speckleRange = 1)         
    
    luf = cv2.undistort(lf, calib_left_mat, calib_left_dist)
    ruf = cv2.undistort(rf, calib_right_mat, calib_right_dist)

    lf_g = cv2.cvtColor(luf, cv2.COLOR_BGR2GRAY)
    rf_g = cv2.cvtColor(ruf, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(lf_g, rf_g)

    display_norm = numpy.zeros(disparity.shape, dtype = "uint8")
    cv2.normalize(disparity,display_norm, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

    cv2.imshow("Disparity", display_norm)

    #detect features in this frame
    detector = cv2.ORB_create()
    lkp, ldesc = detector.detectAndCompute(luf, None)

    keypoints = []
    points_3d = []

    if last_cap is None:
        for k in lkp:
            keypoints.append(decompose_keypoint(k))
    else:
        keypoints, descs = get_similar_feature_keypoints(lkp,ldesc,last_cap[2])        
    
    luf_copy = luf.copy()
    for x,y in keypoints:
        depth = disparity[int(y)][int(x)]
        point_3d = image_plane_to_world_space(x, y, cp_x, cp_y, fx, fy, depth)
        points_3d.append(point_3d)


        r = (255.0 / 150.0) * abs(point_3d[0]) #the closer to center x, the greater r will be
        g = (255.0 / 113) * abs(point_3d[1]) # the closer to center y, the greater g will be 
        b = (255.0 / 1024.0)*(depth) # the closer, the greater b will be 

        cv2.circle(luf_copy,(int(x),int(y)),2,(int(r),int(b),int(g)),5)

    cv2.imshow("Left", luf_copy)


    if(len(points_3d) != 0 and len(keypoints) != 0):
        ret, rvec, tvec = cv2.solvePnP(numpy.asarray(points_3d), numpy.asarray(keypoints), calib_left_mat, distCoeffs= None)


        if last_cap is not None:
            match_frame = cv2.drawMatchesKnn(luf,lkp,last_cap[0],last_cap[1],descs, outImg = None,flags=2)
            cv2.putText(match_frame,str(tvec),(0,20), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),1,cv2.LINE_AA)
            cv2.imshow("Matches", match_frame)

        last_cap = [luf, lkp, ldesc, ret, rvec, tvec]

if __name__=="__main__":
    lvc = cv2.VideoCapture(1)
    rvc = cv2.VideoCapture(0)
    
    while True:
        lret, lf = lvc.read()
        rret, rf = rvc.read()
        
        res = slam(lf, rf)
    
        k = cv2.waitKey(1)
        if k == 27:
            break
