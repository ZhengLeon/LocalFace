import cv2
import dlib
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from skimage.transform import resize


detector = dlib.get_frontal_face_detector()
image = cv2.imread('./test_pic/AF1.jpg')
b, g, r = cv2.split(image)
image_rgb = cv2.merge([r, g, b])
rects = detector(image_rgb, 1)
print(len(rects))
if len(rects) >= 1:
    for rect in rects:
        lefttop_x = rect.left()
        lefttop_y = rect.top()
        rightbottom_x = rect.right()
        rightbottom_y = rect.bottom()
        print(lefttop_x,lefttop_y,rightbottom_x,rightbottom_y)
        cv2.rectangle(image, (lefttop_x, lefttop_y), (rightbottom_x, rightbottom_y), (0, 255, 0), 2)

    cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


