import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

with open('getal.txt') as f:
    img_counter = int(f.readline())
    f.close()
    print(img_counter)

def findColor(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,30,40])
    upper = np.array([30,167,255])
    mask = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow("img", mask)
    box = getContours(mask)
    if box is not None:
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)



def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            cv2.drawContours(imgResult, cnt, -1, (0,255,0),3)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            return box


while True:
    succes, img = cap.read()

    imgResult = img.copy()
    findColor(imgResult)
    cv2.imshow("VIdeo 2", imgResult)










    k = cv2.waitKey(1)
    if k % 256 == 27:
        with open('getal.txt','w') as f:
            f.write(str(img_counter))
            f.close()
        break

    elif k % 256 == 32:
        # SPACE pressed
        img_name = "{}.png".format(img_counter)
        cv2.imwrite(img_name, imgResult)
        print("{} written!".format(img_name))
        img_counter += 1
