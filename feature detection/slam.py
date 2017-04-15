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

depth_string_pixel = ""
depth_string_cm = ""
max_features = 50

point_3d_string = ""
total_position = 0
asdfasdfasdf
DELTA_Z_THRESHOLD = 3

debug_file = open("debug.txt", 'w')

def image_plane_to_world_space(pixel_x, pixel_y, image_center_x, image_center_y, focal_length_x, focal_length_y, depth):
    x = ((pixel_x - image_center_x)*depth_pixel_to_cm(depth)) / (focal_length_x)
    y = ((pixel_y - image_center_y)*depth_pixel_to_cm(depth)) / (focal_length_y)
    return [x,y,depth_pixel_to_cm(depth)]


def decompose_keypoint(keypoint):
    keypoint_out = []

    keypoint_out.append(float(keypoint.pt[0]))
    keypoint_out.append(float(keypoint.pt[1]))

    return keypoint_out

def get_similar_feature_keypoints(keypoints, descs_3d_point_matrix_new, descs_3d_point_matrix_old):
    similar_keypoints_x_y_out = []
    desc_out = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(numpy.asarray(descs_3d_point_matrix_new[0]),numpy.asarray(descs_3d_point_matrix_old[0]), k = 2)

    #Filter poor matches
    average_delta_z = 0
    delta_z_mat = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:

            #find old 3d point
            point_old = descs_3d_point_matrix_old[1][m.trainIdx]
            point_new = descs_3d_point_matrix_new[1][m.queryIdx]

            #calculate change in z
            delta_z = float(point_new[2]) - float(point_old[2])  
            average_delta_z += abs(delta_z)

            #add keypoint to a mat with delta z
            keypoint = keypoints[m.trainIdx]
            delta_z_mat.append([delta_z,keypoint, [m]])


    average_delta_z = average_delta_z / len(matches)
    print "Avg delta z: " + str(average_delta_z)

    #remove keypoints that have changed z too fast (noise)
    for delta_z, keypoint, m in delta_z_mat:
        if abs(delta_z - average_delta_z) < DELTA_Z_THRESHOLD:      
            keypoint_x_y = decompose_keypoint(keypoint)
            similar_keypoints_x_y_out.append(keypoint_x_y)
            desc_out.append(m)

    return (numpy.asarray(similar_keypoints_x_y_out), desc_out)

def slam(lf, rf):
    global last_cap
    global disparity
    global total_position
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
    
    #undistort images
    luf = cv2.undistort(lf, calib_left_mat, calib_left_dist)
    ruf = cv2.undistort(rf, calib_right_mat, calib_right_dist)

    lf_g = cv2.cvtColor(luf, cv2.COLOR_BGR2GRAY)
    rf_g = cv2.cvtColor(ruf, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(lf_g, rf_g)

    display_norm = numpy.zeros(disparity.shape, dtype = "uint8")
    cv2.normalize(disparity,display_norm, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

    cv2.putText(display_norm,depth_string_pixel,(0,20), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(display_norm,depth_string_cm,(0,50), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),1,cv2.LINE_AA)
    cv2.imshow("Disparity", display_norm)

    detector = cv2.ORB_create()

    #detect features in this frame
    lkp, ldesc = detector.detectAndCompute(luf, None)

    #pick out the best features
    if len(lkp) > max_features:
        kpde = list(zip(lkp, ldesc))
        lkpde = sorted(kpde, key=lambda kd: kd[0].response, reverse=True)
        lkpde = lkpde[:max_features]
        lkp, ldesc = list(zip(*lkpde))
        lkp = list(lkp)
        ldesc = numpy.asarray(ldesc)

        
    similar_keypoints_x_y = []
    points_3d = []
    desc_3d_point_matrix = ([],[])

    #stop if there are not enough keypoints
    if len(lkp) > 2:

        #match keypoints with 3d points
        i = 0
        for k in lkp:
            x = float(k.pt[0])
            y = float(k.pt[1])
            depth = disparity[int(y)][int(x)]
            point_3d = image_plane_to_world_space(x, y, cp_x, cp_y, fx, fy, depth)

            desc_3d_point_matrix[0].append(ldesc[i])
            desc_3d_point_matrix[1].append(point_3d)
            i+=1


        if last_cap is None:
            for k in lkp:
                similar_keypoints_x_y.append(decompose_keypoint(k))
        else:
            #get features that are similar between current and last frame
            similar_keypoints_x_y, descs = get_similar_feature_keypoints(lkp,desc_3d_point_matrix,last_cap[2])        
        

        luf_copy = luf.copy()

        for x,y in similar_keypoints_x_y:
            #calculate 3D point for each common feature
            depth = disparity[int(y)][int(x)]
            point_3d = image_plane_to_world_space(x, y, cp_x, cp_y, fx, fy, depth)
            points_3d.append(point_3d)

            #Todo, these values may need recalibrated
            r = (255.0 / 150.0) * abs(point_3d[0]) #the closer to center x, the greater r will be
            g = (255.0 / 113) * abs(point_3d[1]) # the closer to center y, the greater g will be 
            b = (255.0 / 1024.0)*(depth) # the closer, the greater b will be 

            cv2.circle(luf_copy,(int(x),int(y)),2,(int(r),int(b),int(g)),5)

        #draw 3d points
        cv2.putText(luf_copy,point_3d_string,(0,20), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),1,cv2.LINE_AA)
        cv2.imshow("Features in 3D", luf_copy)
    
        #check that there are enough points to get a valid transformation
        if(len(points_3d) > 2  and len(similar_keypoints_x_y) > 2):
            #Object to camera tranformation
            ret, rvec, tvec = cv2.solvePnP(numpy.asarray(points_3d), numpy.asarray(similar_keypoints_x_y), calib_left_mat, calib_left_dist)
            camera_rotation = cv2.Rodrigues(rvec)[0]
            #camera position relative to object
            camera_position = -cv2.transpose(numpy.matrix(camera_rotation)) * numpy.matrix(tvec)

            for mat in tvec:
                debug_file.write(str(mat))
            debug_file.write("\n")

            if last_cap is not None:
                match_frame = cv2.drawMatchesKnn(luf,lkp,last_cap[0],last_cap[1], descs, outImg = None,flags=2)
                cv2.putText(match_frame,str(camera_position),(0,20), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),1,cv2.LINE_AA)
                cv2.imshow("Matches", match_frame)

            #save features for next iteration
        last_cap = [luf, lkp, desc_3d_point_matrix]

        
def callback_depth(event, x, y, flags, params):
     global depth_string_pixel
     global depth_string_cm
     
     depth_pixel = disparity[y][x]
     depth_string_pixel = str(depth_pixel) + " px"
     depth_string_cm = str(depth_pixel_to_cm(depth_pixel)) + " cm"
     
def callback_3d_points(event, x,y,flags, params):
    global point_3d_string
    
    depth = disparity[y][x]
    point = image_plane_to_world_space(x, y, cp_x, cp_y, fx, fy, depth)
    point_3d_string = str((point)) + " (cm)"
    
def depth_pixel_to_cm(pixel_val):
    return 307.0091 + pixel_val*-0.158
    
if __name__=="__main__":
    lvc = cv2.VideoCapture(1)
    rvc = cv2.VideoCapture(0)
    
    cv2.namedWindow("Disparity")
    cv2.setMouseCallback("Disparity", callback_depth)
    
    cv2.namedWindow("Features in 3D")
    cv2.setMouseCallback("Features in 3D", callback_3d_points)
    
    while True:
        lret, lf = lvc.read()
        rret, rf = rvc.read()
        
        res = slam(lf, rf)
    
        k = cv2.waitKey(1)
        if k == 27:
            break
