import cv2 as cv
import numpy as np

from stereovision_calibrate import *
from screenshot_taker import *
from Kalibreren.clicker import *
from RTcalibration import *

# We zoeken K en distortie
dir1 = 'C:/Users/bramo/PycharmProjects/TestEnvProject/venv/images/stereoLeft/*.png'
dir2 = 'C:/Users/bramo/PycharmProjects/TestEnvProject/venv/images/stereoright/*.png'
dir3 = 'C:/Users/bramo/PycharmProjects/TestEnvProject/venv/images/RTleft'
dir4 = 'C:/Users/bramo/PycharmProjects/TestEnvProject/venv/images/RTright'
chessboardsize = (14, 9)
framesize = (640, 480)
sizeOfSq = 17.5
K1, dist1 = findk(dir1, chessboardsize, framesize, sizeOfSq)
K2, dist2 = findk(dir2, chessboardsize, framesize, sizeOfSq)

def draw(img, corners, imgpts):
    def makeTuple(tuppel,index = 0):
        corner = tuple(tuppel[index].ravel())
        corner = np.array(corner)
        corner2 = corner.astype(int)
        corner2 = corner2.astype(tuple)
        return corner2

    yeet = makeTuple(corners)
    img = cv.line(img, yeet, makeTuple(imgpts), (255,0,0), 5)
    img = cv.line(img, yeet, makeTuple(imgpts,1), (0,255,0), 5)
    img = cv.line(img, yeet, makeTuple(imgpts,2), (0,0,255), 5)
    return img
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

#We nemen foto's om te kalibreren en slaan ze op in een mapje
RTCalibrate(dir3,dir4)
imagesLeft = glob.glob('C:/Users/bramo/PycharmProjects/TestEnvProject/venv/images/RTleft/*.png')
imagesRight = glob.glob('C:/Users/bramo/PycharmProjects/TestEnvProject/venv/images/RTright/*.png')

#We willen pixelcoördinaten vegelijken met wereldcoördinaten door te zoeken naar hoeken van een schaakbord
#Wereldcoördinaten initialiseren:
world_points = np.zeros((chessboardsize[0] * chessboardsize[1], 3), np.float32)
world_points[:, :2] = np.mgrid[0:chessboardsize[0], 0:chessboardsize[1]].T.reshape(-1, 2)
world_points = world_points * sizeOfSq
imgpointsL = []
imgpointsR = []
objpoints = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
axis = np.float32([[3*sizeOfSq, 0, 0], [0, 3*sizeOfSq, 0], [0, 0, -3*sizeOfSq]]).reshape(-1, 3)

#We overlopen nu de verschillende foto's om zo R en T te bepalen
for imgLeft,imgRight in zip(imagesLeft,imagesRight):
    print('yeet')
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)

    # Find the rotation and translation vectors.
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardsize,None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardsize, None)

    #Als beide afbeeldingen de hoeken vinden, voeren we de code verder uit
    if retL and retR:
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(imgL, chessboardsize, cornersL, retL)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(imgR, chessboardsize, cornersR, retR)

        cv.imshow('img left', imgL)
        cv.imshow('img left', imgR)
        cornersL_ = allignPoints(cornersL[0][0], cornersL[1][0], cornersL[-1][0], cornersL)
        cornersR_ = allignPoints(cornersR[0][0], cornersR[1][0], cornersR[-1][0], cornersR)
        if cornersL_ is not None and cornersR_ is not None:
            cv.waitKey(100)

            imgpointsL.append(cornersL_)
            imgpointsR.append(cornersR_)
            objpoints.append(world_points)
        cv.destroyAllWindows()


#Nu berekenen we R en T met de solvePnp methode
retL, cameramatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, framesize, None, None,)
heightL,widthL, channels = imgL.shape
newcameramatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameramatrixL, distL, (widthL, heightL), 0, (widthL, heightL))
#imgptsL, jac = cv.projectPoints(axis, rvecsL, tvecsL, K1, dist1)
#img = draw(grayL, cornersL_, imgptsL)
#cv.imshow('img left', img)
#k = cv.waitKey(0) & 0xFF

retR, cameramatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, framesize, None, None,)
heightR,widthR, channels = imgL.shape
newcameramatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameramatrixR, distR, (widthR, heightR), 0, (widthR, heightR))
#imgptsR, jac = cv.projectPoints(axis, rvecsR, tvecsR, K2, dist2)
#img = draw(img, cornersR_, imgptsR)
#cv.imshow('img left', img)
#k = cv.waitKey(0) & 0xFF
#We gaan uit van een stereocamera aangezien dit het rekenwerk veel makkelijker maakt

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
#We linken de twee camera's aan elkaar
retStereo, newcameramatrixL,distL, newcameramatrixR,distR,rot,trans, essentialMatrix, fundamentalMatrix =cv.stereoCalibrate(objpoints,imgpointsL,imgpointsR,K1,dist1,K2,dist2,grayL.shape)

#Nu zorgen we dat de camera's op dezelfde lijn staan en dezelfde kant opkijken om zo een 'echte' stereocamera te bekomen
rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR,Q,roi_L, roi_R = cv.stereoRectify(newcameramatrixL,distL,newcameramatrixR,distR, grayL.shape[::-1],rot,trans, rectifyScale, (0,0))

stereoMapL = cv.initUndistortRectifyMap(newcameramatrixL,distL,rectL,projMatrixL,grayL.shape[::-1],cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newcameramatrixR,distR,rectR,projMatrixR,grayR.shape[::-1],cv.CV_16SC2)

print("Saving parameters!")
np.save('K1.npy',newcameramatrixL, allow_pickle=True)
print(newcameramatrixL)
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()

