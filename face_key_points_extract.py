import cv2
import dlib
import numpy as np 


def key_points(img):
    points_keys = []
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    rects = detector(img, 1)

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rects[i]).parts()])
        img = img.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0,0],point[0,1])
            points_keys.append(pos)
            cv2.circle(img,pos,2,(255,0,0),-1)
    return img

frame=cv2.imread('./test_pic/b.png')
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_key = key_points(frame)
cv2.imshow('frame',face_key)
cv2.waitKey(10000) 

