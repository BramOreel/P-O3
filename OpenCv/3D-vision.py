import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


#Script die filtert op rood bvb en mij de coör doorgeeft

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
K1 = np.load('Kalibreren/K1.npy', allow_pickle=True)
frame_rate = 30    #Camera frame rate (maximum at 30 fps)
B = 7              #Distance between the cameras [cm]
f = 6               #Camera lense's focal length [mm]
alpha = 55
#Camera field of view in the horisontal plane [degrees]
cap1.set(3,640)
cap1.set(4,480)
cap2.set(3,640)
cap2.set(4,480)

x = np.array([])
y = np.array([])
z = np.array([])

def findColor(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([155,0,0])
    upper = np.array([179,255,255])
    mask = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow("img", mask)
    return mask
   # if c is not None:
      #  cv2.circle(img, (int(c[0]), int(c[1])), radius=20, color=(0, 0, 255), thickness=4)

def getContours(mask,imgResult):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursSrtd = sorted(contours, key=lambda x: cv2.contourArea(x))
    if contoursSrtd:
        area = cv2.contourArea(contoursSrtd[-1])
        if area > 500:
            cv2.drawContours(imgResult, contoursSrtd[-1], -1, (0, 255, 0), 3)
            c, w, a = cv2.minAreaRect(contoursSrtd[-1])
            return c

def find_depth(right_point, left_point, frame_right, frame_left, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity     #Depth in [cm]
    xDepth = (x_right*baseline)/disparity
    yDepth = ((right_point[1]+left_point[1])*baseline)/(2*disparity)
    return xDepth,yDepth,zDepth

def getXml(frameR, frameL,stereoMapL_x,stereoMapL_y,stereoMapR_x,stereoMapR_y):


    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    return undistortedR, undistortedL


while True:

    ret_left, frame_left = cap1.read()
    ret_right, frame_right = cap2.read()
    imgResult = frame_left.copy()
    imgResult2 = frame_right.copy()

    cv_file = cv2.FileStorage()
    cv_file.open('Kalibreren/stereoMap.xml', cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    frame_right, frame_left = getXml(frame_right, frame_left,stereoMapL_x,stereoMapL_y,stereoMapR_x,stereoMapR_y)

    if not ret_left or not ret_right:
        break

    else:

        start = time.time()

        mask = findColor(frame_left)
        c1 = getContours(mask, frame_left)
        if c1 is not None:
            cv2.circle(imgResult, (int(c1[0]), int(c1[1])), radius=20, color=(0, 0, 255), thickness=4)


        mask2 = findColor(frame_right)
        c2 = getContours(mask2, frame_right)
        if c2 is not None:
            cv2.circle(imgResult2, (int(c2[0]), int(c2[1])), radius=20, color=(0, 0, 255), thickness=4)



        if c1 is not None and c2 is not None:
            depthx,depthy,depthz = find_depth(c2, c1, frame_right, frame_left, B, f, alpha)

            #cv2.putText(frame_right, "Distance: " + str(round(depthz, 1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        #(0, 255, 0), 3)

            #cv2.putText(frame_left, "Distance: " + str(round(depthz, 1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        #(0, 255, 0), 3)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            print('X: '+str(depthx) +'\tY: ' +str(depthy)+ '\tZ' + str(depthz) + '\n')

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            # print("FPS: ", fps)

            cv2.putText(frame_right, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame_left, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Show the frames
        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)

            # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release and destroy all windows before termination
cap1.release()
cap2.release()

cv2.destroyAllWindows()















