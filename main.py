import cv2
import dlib
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from skimage.transform import resize


detector = dlib.get_frontal_face_detector()
image = cv2.imread('./test_pic/b.png')
rects = detector(image, 1)

def key_points(img):
    points_keys = []
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rects[i]).parts()])
        img = img.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0,0],point[0,1])
            points_keys.append(pos)
            cv2.circle(img,pos,2,(255,0,0),-1)
    return img


if len(rects) >= 1:
    for rect in rects:
        lefttop_x = rect.left()
        lefttop_y = rect.top()
        rightbottom_x = rect.right()
        rightbottom_y = rect.bottom()
        cv2.rectangle(image, (lefttop_x, lefttop_y), (rightbottom_x, rightbottom_y), (0, 255, 0), 2)
    face_key = key_points(image)
    cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('result', face_key)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


