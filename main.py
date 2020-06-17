import cv2
import dlib
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from skimage.transform import resize


# image = cv2.imread('./test_pic/AF1.jpg')
# image = cv2.imread('./test_pic/AF1675.jpg')
# image = cv2.imread('./test_pic/a.png')
# image = cv2.imread('./test_pic/b.png')
image = cv2.imread('./test_pic/c.png')
detector = dlib.get_frontal_face_detector()
rects = detector(image, 1)

def key_points(img):
    points_keys = []
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # for i in range(len(rects)):
    # len(rects)代表人脸个数
    landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rects[0]).parts()])
    maxv = np.amax(landmarks, axis=0)
    minv = np.amin(landmarks, axis=0)
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0,0],point[0,1])
        points_keys.append(pos)
        cv2.circle(img,pos,2,(255,0,0),-1)
    return img, maxv, minv


face_key, maxv, minv = key_points(image)
lefttop_x = minv.A[0][0]
lefttop_y = minv.A[0][1]
rightbottom_x = maxv.A[0][0]
rightbottom_y = maxv.A[0][1]
# print(lefttop_x,lefttop_y,rightbottom_x,rightbottom_y)
cv2.rectangle(face_key, (lefttop_x, lefttop_y), (rightbottom_x, rightbottom_y), (0, 255, 0), 2)

cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
cv2.imshow('result', face_key)
cv2.waitKey(0)
cv2.destroyAllWindows()


