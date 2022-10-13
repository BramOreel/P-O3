import cv2


cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap.set(3,640)
cap.set(4,480)
cap2.set(3,640)
cap2.set(4,480)

while True:
    ret0, img = cap.read()
    ret1, img2 = cap2.read()

    cv2.imshow("VIdeo 2", img)
    cv2.imshow("Video", img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break