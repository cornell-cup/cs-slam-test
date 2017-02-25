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
    left = np.load("configuration/cam_mats_left.npy")
    right = np.load("configuration/cam_mats_right.npy")

    return left, right

def load_distance_coeffs():
    left = np.load("configuration/dist_coefs_left.npy")
    right = np.load("configuration/dist_coefs_right.npy")

    return left, right

def get_depth(img_1, img_2):
    """ Takes 2 images captured from a stereo vision rig and computes the disparities between the image.

    Args:
        img_1 - first image
        img_2 - second image
    
    Returns:
        A np array containing 16 bit fixed point values representing a disparty map
    """

    stereo = cv2.StereoSGBM_create(minDisparity = 16,
        numDisparities = 96,
        blockSize = 16,
        P1 = 8*3*3*2,
        P2 = 32*3*3*2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32)

    disparity = stereo.compute(img_2, img_1).astype(np.float32) / 16.0

    h, w = img_1.shape[:2]
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
    #print('%s saved' % 'out.ply')

    return disparity

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
    cv2.namedWindow("undistorted_left")
    cv2.namedWindow("undistorted_right")

    while True:
        img_left = cv2.imread("left_capture.jpg")
        img_right = cv2.imread("right_capture.jpg")

        left_camera_matrix, right_camera_matrix         = load_camera_matrix()
        left_camera_dist_coefs, right_camera_dist_coefs = load_distance_coeffs()

        left_camera_matrix[0][0] = left_camera_matrix[0][0]*1.5
        left_camera_matrix[0][2] = left_camera_matrix[0][2]*1.5
        left_camera_matrix[1][1] = left_camera_matrix[1][1]*1.125
        left_camera_matrix[1][2] = left_camera_matrix[1][2]*1.125

        right_camera_matrix[0][0] = right_camera_matrix[0][0]*1.5
        right_camera_matrix[0][2] = right_camera_matrix[0][2]*1.5
        right_camera_matrix[1][1] = right_camera_matrix[1][1]*1.125
        right_camera_matrix[1][2] = right_camera_matrix[1][2]*1.125

        img_left_undistorted = cv2.undistort(img_left,   left_camera_matrix,  left_camera_dist_coefs)
        img_right_undistorted = cv2.undistort(img_right, right_camera_matrix, right_camera_dist_coefs)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

        if img_left is not None and img_right is not None:
            depth_image = None
            #depth_image = get_depth(img_left,img_right)
            if depth_image is not None:
                cv2.imshow("depth", (depth_image-16)/96)

            cv2.imshow("left", img_left)
            cv2.imshow("right", img_right)
            cv2.imshow("undistorted_left",img_left_undistorted)
            cv2.imshow("undistorted_right",img_right_undistorted)

if __name__ == "__main__":
    loop()
