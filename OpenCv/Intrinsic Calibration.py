import numpy as np
import cv2 as cv
import glob

def allignPoints(p1,p2,pn,corners):
    if abs(p2[0] - p1[0]) > abs(p2[1] - p1[1]): #horizontaal
        print('Horizontaal' + '\t')
        if p1[0] < pn[0]: #linksboven of linksonder
            if p1[1] < pn[1]: #linksboven
                print('linksboven')

            else: #linksonder
                print('linksonder')
                return None

        else: #rechtsboven of rechtsonder
            if p1[1] < pn[1]: #rechtsboven
                print('rechtsboven')
                return None

            else: #rechtsonder
                corners = corners[::-1]
                print('rechtsonder')

    else: #verticaal
        return None
    return corners

def findk(direc, chessboardsize, framesize, sizeofsq):

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((14 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:14, 0:9].T.reshape(-1, 2)
    objp = objp * sizeofsq
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = sorted(glob.glob(direc))
    for img in images:

        img_x = cv.imread(img)
        gray = cv.cvtColor(img_x, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardsize, flags = cv.CALIB_CB_MARKER)
        # If found, add object points, image points (after refining them)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print(corners[0], corners[1], corners[-1])
            corners2 = allignPoints(corners[0][0], corners[1][0], corners[-1][0], corners)
            if corners2 is not None:
                print(corners2[0] , corners2[1] , corners2[-1])


                objpoints.append(objp)
                imgpoints.append(corners)
        # Draw and display the corners
                cv.drawChessboardCorners(img_x, chessboardsize, corners, ret)
                cv.imshow(img, img_x)
                cv.waitKey(100)
                cv.destroyAllWindows()

    # CALIBRATION

    ret, cameramatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, framesize, None, None,)
    height,width, channels = img_x.shape
    newcameramatrix, roi = cv.getOptimalNewCameraMatrix(cameramatrix, dist, (width, height), 0, (width, height))

    mapx, mapy = cv.initUndistortRectifyMap(newcameramatrix, dist, None, newcameramatrix, (width, height), 5)

    #src = 'C:/Users/bramo/PycharmProjects/TestEnvProject/Kalibreren/screenshot0.png'
    #src = cv.imread(src)
    #src = cv.remap(src, mapx, mapy, cv.INTER_LINEAR)
#
 #   x, y, w, h = roi
  #  src = src[y:y + h, x:x + w]
#
    #cv.imshow('newCameramatrix', src)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return newcameramatrix, dist
