import cv2
import os
import glob
import dlib
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from skimage.transform import resize

i = 0
root_path = "C:/Users/abc/Desktop/Global_Local_FBP/Images/AF1183.jpg"

image = cv2.imread(root_path)
detector = dlib.get_frontal_face_detector()
rects = detector(image, 1)

def key_points(img):
    points_keys = []
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rects[0]).parts()])
    maxv = np.amax(landmarks, axis=0)
    minv = np.amin(landmarks, axis=0)
    return img, maxv, minv
if len(rects) > 0:
    face_key, maxv, minv = key_points(image)
    lefttop_x = minv.A[0][0]
    lefttop_y = minv.A[0][1]
    rightbottom_x = maxv.A[0][0]
    rightbottom_y = maxv.A[0][1]
    face = face_key[lefttop_y: rightbottom_y, lefttop_x: rightbottom_x] 
    cv2.imwrite('test.jpg',face)
else:
    print(i, "no face")

i = i + 1
print(i)





