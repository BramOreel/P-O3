import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap.set(3,640)
cap.set(4,480)
cap2.set(3,640)
cap2.set(4,480)

faceCascade = cv2.CascadeClassifier("C:/Users/bramo/PycharmProjects/OpenCv/haarcascade_frontalface_default.xml")

while True:
    succes, img = cap.read()
    succes2, img2 = cap2.read()
    Imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Imgray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(Imgray, 6,3)
    faces2 = faceCascade.detectMultiScale(Imgray2, 6, 3)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0))

    for (x,y,w,h) in faces2:
        cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0))

    cv2.imshow("VIdeo 2", img)
    cv2.imshow("Video", img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


