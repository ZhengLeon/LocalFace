import cv2
import dlib
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from skimage.transform import resize


image = cv2.imread('./test_pic/AF1.jpg')
detector = dlib.get_frontal_face_detector()
rects = detector(image, 1)

def key_points(img):
    points_keys = []
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rects[i]).parts()])
        maxv = np.amax(landmarks, axis=0)
        minv = np.amin(landmarks, axis=0)
    return img, maxv, minv


face_key, maxv, minv = key_points(image)
lefttop_x = minv.A[0][0]
lefttop_y = minv.A[0][1]
rightbottom_x = maxv.A[0][0]
rightbottom_y = maxv.A[0][1]
# cv2.rectangle(face_key, (lefttop_x, lefttop_y), (rightbottom_x, rightbottom_y), (0, 255, 0), 2)
face = face_key[lefttop_y: rightbottom_y, lefttop_x: rightbottom_x] 
cv2.imwrite('./face/AF1_face.jpg',face)



