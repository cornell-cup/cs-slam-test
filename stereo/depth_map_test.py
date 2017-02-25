import cv2
import numpy as np

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def load_camera_matrix():
    left = np.load("calibration/cam_mats_left.npy")
    right = np.load("calibration/cam_mats_right.npy")

    return left, right

def load_distance_coeffs():
    left = np.load("calibration/dist_coefs_left.npy")
    right = np.load("calibration/dist_coefs_right.npy")

    return left, right

def get_depth(img_1, img_2):
    """ Takes 2 images captured from a stereo vision rig and computes the disparities between the image.

    Args:
        img_1 - first image
        img_2 - second image
    
    Returns:
        A np array containing 16 bit fixed point values representing a disparty map
    """

    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)


    SADWindowSize = 20
    p1 = 8 * 1 * SADWindowSize * SADWindowSize
    p2 = 4 * p1

    stereo = cv2.StereoSGBM_create(
        minDisparity = -21,
        numDisparities = 96,
        blockSize = 9,
        P1 = p1,
        P2 = p2,
        disp12MaxDiff = 1,
        preFilterCap = 63,
        uniquenessRatio = 7,
        speckleWindowSize = 50,
        speckleRange = 1)

    disparity = stereo.compute(img_1, img_2)

    disparity_visual = np.zeros(disparity.shape, dtype = "uint8")
    cv2.normalize(disparity, disparity_visual, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    '''h, w = img_1.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])

    points = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    #write_ply('out.ply', out_points, out_colors)
    #print('%s saved' % 'out.ply')'''

    return disparity_visual

def loop():
    vc_1 = cv2.VideoCapture(0)
    vc_2 = cv2.VideoCapture(1)

    vc_1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
    vc_1.set(cv2.CAP_PROP_FPS, 30)
    vc_1.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    vc_2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vc_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
    vc_2.set(cv2.CAP_PROP_FPS, 30)
    vc_2.set(cv2.CAP_PROP_AUTOFOCUS, 0)


    cv2.namedWindow("left")
    cv2.namedWindow("right")
    cv2.namedWindow("undistorted_left")
    cv2.namedWindow("undistorted_right")

    while True:

        _, img_right = vc_1.read()
        _, img_left = vc_2.read()

        #img_left = cv2.imread("left_capture.jpg")
        #img_right = cv2.imread("right_capture.jpg")

        left_camera_matrix, right_camera_matrix         = load_camera_matrix()
        left_camera_dist_coefs, right_camera_dist_coefs = load_distance_coeffs()

        img_left_undistorted = cv2.undistort(img_left,   left_camera_matrix,  left_camera_dist_coefs)
        img_right_undistorted = cv2.undistort(img_right, right_camera_matrix, right_camera_dist_coefs)

        cv2.imwrite("test_left.jpg",img_left_undistorted)
        cv2.imwrite("test_right.jpg",img_right_undistorted)

        break
        '''print "left dist: " + str(left_camera_dist_coefs)
        print "right dist:" + str(right_camera_dist_coefs)

        print "left mat: " + str(left_camera_matrix)
        print "right mat: " + str(right_camera_matrix)
        
        break'''
        key = cv2.waitKey(1)
        if key == 27:
            break

        if img_left is not None and img_right is not None:
            depth_image = get_depth(img_left_undistorted,img_right_undistorted)
            if depth_image is not None:
                cv2.imshow("depth", depth_image)

            cv2.imshow("left", img_left)
            cv2.imshow("right", img_right)
            cv2.imshow("undistorted_left",img_left_undistorted)
            cv2.imshow("undistorted_right",img_right_undistorted)

if __name__ == "__main__":
    loop()
