import cv2
import numpy as np


def findColor(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([15,0,86])
    upper = np.array([179,103,255])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    frame1_ = findColor(frame1)
    frame2_ = findColor(frame2)

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 15, 200, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    cv2.imshow("dilated",dilated)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

    if cntsSorted and cv2.contourArea(cntsSorted[-1]) > 400:
        cv2.drawContours(frame1 , cntsSorted[-1], -1, (0,255,0),2)
        rect = cv2.minAreaRect(cntsSorted[-1])
        if(rect[1][0]> rect[1][1]):
            print(int(rect[2]))
        else:
            print(90 - rect[2])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(frame1, [box], 0, (0, 0, 255), 2)

    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()

