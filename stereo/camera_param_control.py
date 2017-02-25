import cv2

cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
cap1.set(cv2.CAP_PROP_FPS, 30)

cap1.set(CV_CAP_PROP_SETTINGS, 1);
