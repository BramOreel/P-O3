import time
import numpy as np
import cv2


K1 = np.load('K1.npy', allow_pickle=True)
frame_rate = 60         #Camera frame rate (maximum at 60 fps)
B = #invullen           #Distance between the cameras [cm]
f = K1[0][0]            #Camera lense's focal length [mm]
alpha = 90

cap1.set(3,640) #Deze op de grootte van je frame zetten
cap1.set(4,480)
cap2.set(3,640)
cap2.set(4,480)

cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

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
    zDepth = (baseline*f_pixel)/-disparity     #Depth in [cm]
    xDepth = (x_right*baseline)/-disparity
    yDepth = ((right_point[1]+left_point[1])*baseline)/(-2*disparity)
    return xDepth,yDepth,zDepth
def getXml(frameR, frameL,stereoMapL_x,stereoMapL_y,stereoMapR_x,stereoMapR_y):


    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    return undistortedR, undistortedL


while True:

    start = time.time()
    frame_right, frame_left = getXml(frame_right, frame_left,stereoMapL_x,stereoMapL_y,stereoMapR_x,stereoMapR_y)

    c1 = #tupel met coör van linkerframe
    c2 = #tupel met coör van rechterframe
    if not ret_left or not ret_right: #ret bij cap.read van links en rechts
        break

    else:
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

            #cv2.putText(frame_right, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            #cv2.putText(frame_left, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)