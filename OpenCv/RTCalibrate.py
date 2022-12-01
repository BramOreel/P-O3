import cv2
import os

def RTCalibrate(dir1,dir2):
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    #dir1 = 'C:/Users/bramo/PycharmProjects/TestEnvProject/venv/images/stereoLeft'
    #dir2 = 'C:/Users/bramo/PycharmProjects/TestEnvProject/venv/images/stereoright'

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)

    os.chdir(dir1)
    num = 0

    while cap.isOpened():

        succes1, img = cap.read()
        succes2, img2 = cap2.read()

        k = cv2.waitKey(5)

        if k == 27:

            return None
        elif k == 32: # wait for 's' key to save and exit
            cv2.imwrite('imageL' + str(num) + '.png', img)
            os.chdir(dir2)
            cv2.imwrite('imageR' + str(num) + '.png', img2)
            os.chdir(dir1)
            print("images saved!")
            num += 1

        cv2.imshow('links',img)
        cv2.imshow('rechts',img2)

# Release and destroy all windows before termination
    cap.release()
    cap2.release()

    cv2.destroyAllWindows()