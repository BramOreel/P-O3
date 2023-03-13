import cv2
import numpy as np
from tracker import *
import matplotlib.pyplot as plt
import scipy.stats as stats
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
import operator





cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Setup
tracker1 = EuclideanDistTracker()
tracker2 = EuclideanDistTracker()

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_EXPOSURE, -60000)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2.set(cv2.CAP_PROP_EXPOSURE, -60000)

cap1.set(3,640) #Deze op de grootte van je frame zetten
cap1.set(4,480)
cap2.set(3,640)
cap2.set(4,480)

ret11, frame11 = cap1.read()
ret12, frame12 = cap1.read()
ret21, frame21 = cap2.read()
ret22, frame22 = cap2.read()

frame11 = cv2.remap(frame11, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
frame12 = cv2.remap(frame12, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
frame21 = cv2.remap(frame21, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
frame22 = cv2.remap(frame22, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


K1 = np.load('K1.npy', allow_pickle=True)
frame_rate = 60         #Camera frame rate (maximum at 60 fps)
B = 159.1             #invullen           #Distance between the cameras [cm]
f = K1[0][0]            #Camera lense's focal length [mm]
alpha = 90




def findColor(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,62,37])
    upper = np.array([65,255,255])
    mask = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow("img", mask)
    return mask
   # if c is not None:
      #  cv2.circle(img, (int(c[0]), int(c[1])), radius=20, color=(0, 0, 255), thickness=4)

def getContours(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contoursSrtd = sorted(contours, key=lambda x: cv2.contourArea(x))
    return contoursSrtd



def vindDetections(diff, frame1, kant):
    detections = []

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 15, 200, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

    for n in range(3):
        if cntsSorted:
            if len(cntsSorted) >= n + 1 and cv2.contourArea(cntsSorted[-n - 1]) > 400:
                cnt = cntsSorted[-n - 1]
                xbox, ybox, wbox, hbox = cv2.boundingRect(cnt)
                crop_img = frame1[ybox:ybox + hbox, xbox:xbox + wbox]
                cntsKleur = getContours(findColor(crop_img))
                if cntsKleur:
                    rect = cv2.minAreaRect(cntsKleur[-1])
                    x, y, w, h = round(rect[0][0]), round(rect[0][1]), rect[1][0], rect[1][1]
                    if w > h:
                        hoek = rect[2]
                    else:
                        hoek = 90 - rect[2]  # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(points)
                    detections.append([x + xbox, y + ybox, h, hoek])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    box = np.add(box,np.array([[xbox, ybox],[xbox, ybox],[xbox, ybox],[xbox, ybox]]))
                    cv2.drawContours(frame1, [box], 0, (0, 0, 255), 2)
    return detections


def vindStok(alleTrajecten):
    alle_x = {}
    alle_y = {}
    alle_w = {}
    alle_h = {}
    alle_frameNrs = {}
    frameNr = 0
    for frame in alleTrajecten:
        for object in frame:
            if not object in alle_x.keys():
                alle_x[object] = []
                alle_y[object] = []
                alle_w[object] = []
                alle_h[object] = []
                alle_frameNrs[object] = []
            x, y, w, h = frame[object]
            alle_x[object].append(x)
            alle_y[object].append(y)
            alle_w[object].append(w)
            alle_h[object].append(h)
            alle_frameNrs[object].append(frameNr)
        frameNr += 1

    alleRes = []
    for object in alle_x:  # verkeerde objecten uitfilteren
        x, y = np.array(alle_x[object]), np.array(alle_y[object])
        if len(x) < 6:
            alleRes.append(math.inf)
        else:
            if max(x) - min(x) < 200:
                alleRes.append(math.inf)
            else:
                flagStijgend = 0
                flagDalend = 0
                i = 2
                while i < len(x) - 1:  # buitenste punten niet meenemen
                    if (x[i] < x[i - 1] - 10):
                        flagStijgend = 1
                    if (x[i] > x[i - 1] + 10):
                        flagDalend = 1
                    i += 1
                if flagStijgend and flagDalend and False:
                    alleRes.append(math.inf)
                else:
                    p, res, _, _, _ = np.polyfit(x, y, 2, full=True)
                    if p[0] > 0:
                        alleRes.append(res)
                    else:
                        alleRes.append(math.inf)

    beste_object = alleRes.index(min(alleRes))

    x, y = np.array(alle_x[beste_object]), np.array(alle_y[beste_object])
    p, res, _, _, _ = np.polyfit(x, y, 2, full=True)
    mymodel = np.poly1d(p)

    myline = np.linspace(min(x), max(x), 100)
    print(alleRes)
    plt.scatter(x, y)
    plt.plot(myline, mymodel(myline))

    plt.plot(x, y)
    plt.show()
    return alle_x[beste_object], alle_y[beste_object], alle_w[beste_object], alle_h[beste_object], alle_frameNrs[
        beste_object]


alleTrajecten1 = []
alleTrajecten2 = []

while cap1.isOpened():
    diff1 = cv2.absdiff(frame11, frame12)
    diff2 = cv2.absdiff(frame21, frame22)

    detections1 = vindDetections(diff1, frame11,1)
    detections2 = vindDetections(diff2, frame21,2)
    boxes_ids1 = tracker1.update(detections1)
    boxes_ids2 = tracker2.update(detections2)
    x_y_w_h1 = {}
    for object in tracker1.center_points:
        x, y = tracker1.center_points[object]
        w, h = tracker1.breedte_hoogte[object]
        x_y_w_h1[object] = (x, y, w, h)
    alleTrajecten1.append(x_y_w_h1)
    x_y_w_h2 = {}
    for object in tracker2.center_points:
        x, y = tracker2.center_points[object]
        w, h = tracker2.breedte_hoogte[object]
        x_y_w_h2[object] = (x, y, w, h)
    alleTrajecten2.append(x_y_w_h2)

    for box_id in boxes_ids1:
        x, y, w, h, id = box_id
        cv2.putText(frame11, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    for box_id in boxes_ids2:
        x, y, w, h, id = box_id
        cv2.putText(frame21, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("feedLinks", frame11)
    frame11 = frame12
    ret1, frame12 = cap1.read()
    frame12 = cv2.remap(frame12, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    cv2.imshow("feedRechts", frame21)
    frame21 = frame22
    ret2, frame22 = cap2.read()
    frame22 = cv2.remap(frame22, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    if cv2.waitKey(5) == 27:
        break


def coorVoorBram(traject1,traject2):
    alle_x1, alle_y1, alle_h1, alle_hoek1, alle_frames1 = traject1
    alle_x2, alle_y2, alle_h2, alle_hoek2, alle_frames2 = traject2
    Coor11 = []
    Coor12 = []
    Coor21 = []
    Coor22 = []
    for index1, frame in enumerate(alle_frames1):
        if frame in alle_frames2:
            index2 = alle_frames2.index(frame)
            x1, y1, h1, hoek1 = alle_x1[index1], alle_y1[index1], alle_h1[index1], alle_hoek1[index1]
            x2, y2, h2, hoek2 = alle_x2[index2], alle_y2[index2], alle_h2[index2], alle_hoek2[index2]
            cos1, sin1 = np.cos(hoek1), np.sin(hoek1)
            cos2, sin2 = np.cos(hoek2), np.sin(hoek2)
            Coor11.append((x1 + cos1 * h1 / 2, y1 + sin1 * h1 / 2))
            Coor12.append((x1 - cos1 * h1 / 2, y1 - sin1 * h1 / 2))
            Coor21.append((x2 + cos2 * h2 / 2, y2 + sin2 * h2 / 2))
            Coor22.append((x2 - cos2 * h2 / 2, y2 - sin2 * h2 / 2))
    return Coor11, Coor12, Coor21, Coor22

"""
Deel van Bram hieronder

"""


def find_depth(right_point, left_point, baseline, alpha):
    f_pixel = (640 * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

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



"""
deel van Jonas:
"""


def uiteindenBepalen(x,y,z):
    uiteinden=[]
    for i in range(len(x)):
        uiteinden.append((x[i],y[i],z[i]))
    return uiteinden


def middelpuntcoordinaten(uiteinden):
    middelpunten = []

    for i in range(len(uiteinden)):
        if i % 2 == 0:
            x = (uiteinden[i][0] + uiteinden[i + 1][0]) / 2
            y = (uiteinden[i][1] + uiteinden[i + 1][1]) / 2
            z = (uiteinden[i][2] + uiteinden[i + 1][2]) / 2
            middelpunt = (x, y, z)
            middelpunten.append(middelpunt)
    return middelpunten


def xyz(middelpunten):
    x = []
    y = []
    z = []
    for i in middelpunten:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    return x, y, z


def puntenOpRechte(lengtedeelgebieden, n):
    nieuwePunten = [0]
    k = 1
    for i in range(n - 1):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1
    return nieuwePunten


def vlakplot(uiteinden, n, slope, intercept, fittedParameters, ax):
    middelpunten = []
    aantal = len(uiteinden) // (2 * n)
    for m in range(aantal):
        if n % 2 == 0:
            punt = tuple(map(operator.add, uiteinden[m * 2 * n + n], uiteinden[m * 2 * n + n + 1]))
        else:
            punt = tuple(map(operator.add, uiteinden[m * 2 * n + n - 1], uiteinden[m * 2 * n + n]))
        middelpunt = tuple(t / 2 for t in punt)
        x = middelpunt[0]
        y = slope * x + intercept
        xy = x**2+y**2
        z = xy * fittedParameters[0] + fittedParameters[1] * np.sqrt(x**2+y**2) + fittedParameters[2]
        #z = middelpunt[2]
        middelpunten.append((x, y, z))

    gebieden = []
    l = 0
    for k in range(aantal):
        punten = [middelpunten[k]]
        for j in range(n):
            oorsprong = tuple(map(operator.sub, uiteinden[l], uiteinden[l + 1]))
            punten.append(tuple(map(operator.add, oorsprong, middelpunten[k])))
            l += 2
        gebieden.append(punten)

    for i in range(len(gebieden)):
        points = Points(gebieden[i])
        plane = Plane.best_fit(points)
        plane.plot_3d(ax, alpha=0.2, lims_x=(-25, 25), lims_y=(-25, 25))
        plane.point.plot_3d(ax, s=30)


def predictparaboolbaanUiteindePunten(middelpunten, n):
    x = []
    y = []
    z = []

    x1 = []
    y1 = []
    z1 = []

    x2 = []
    y2 = []
    z2 = []

    for i in range(n // 2):
        x.append(middelpunten[i][0])
        y.append(middelpunten[i][1])
        z.append(middelpunten[i][2])

        x1.append(middelpunten[i][0])
        y1.append(middelpunten[i][1])
        z1.append(middelpunten[i][2])

    for i in range(n // 2):
        x.append(middelpunten[-i][0])
        y.append(middelpunten[-i][1])
        z.append(middelpunten[-i][2])

        x2.append(middelpunten[-i][0])
        y2.append(middelpunten[-i][1])
        z2.append(middelpunten[-i][2])

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    nieuwePunten = [0]

    yBegin = x1[0] * slope + intercept
    yEind = x1[-1] * slope + intercept
    lengte = yEind - yBegin
    breedte = x1[-1] - x1[0]

    lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
    lengtedeelgebieden = lengtegebied / (len(x1) - 1)

    k = 1
    for i in range(len(x1) - 1):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1

    yBegin = x2[0] * slope + intercept
    yEind = x2[-1] * slope + intercept
    lengte = yEind - yBegin
    breedte = x2[-1] - x2[0]

    lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
    lengtedeelgebieden = lengtegebied / (len(x2) - 1)

    k = 1
    for i in range(len(x2)):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1


    #model = np.poly1d(np.polyfit(nieuwePunten, z, 2))

    fittedParameters = np.polyfit(nieuwePunten, z, 2)

    kwadraat = [a * b for a, b in zip(nieuwePunten, nieuwePunten)]
    z = [a * fittedParameters[0] + fittedParameters[1] * b + fittedParameters[2] for a, b in zip(kwadraat, nieuwePunten)]

    ax.plot(x, y, z)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Interpolerende parabool op basis van de uiteindepunten.")
    plt.show()


def predictparaboolbaanMiddenpunten(middelpunten, n):
    x = []
    y = []
    z = []
    for i in range(n // 4, 3 * n // 4):
        x.append(middelpunten[i][0])
        y.append(middelpunten[i][1])
        z.append(middelpunten[i][2])

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)


    slope, intercept, r, p, std_err = stats.linregress(x, y)

    yEind = x[-1] * slope + intercept
    yBegin = x[0] * slope + intercept
    lengte = yEind - yBegin
    breedte = x[-1] - x[0]

    lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
    lengtedeelgebieden = lengtegebied / (len(x) - 1)

    nieuwePunten = [0]
    k = 1
    for i in range(len(x) - 1):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1

    fittedParameters = np.polyfit(nieuwePunten, z, 2)

    kwadraat = [a * b for a, b in zip(nieuwePunten, nieuwePunten)]
    z = [a * fittedParameters[0] + fittedParameters[1] * b + fittedParameters[2] for a, b in zip(kwadraat, nieuwePunten)]

    ax.plot(x, y, z)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.title("Interpolerende parabool op basis van de middelpunten.")

    plt.show()


def punten(uiteinden, ax):
    k=0
    p = ['red', 'blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(int(len(uiteinden)/2)):
        x = [uiteinden[2*k][0],uiteinden[2*k+1][0]]
        y = [uiteinden[2*k][1], uiteinden[2*k + 1][1]]
        z = [uiteinden[2*k][2], uiteinden[2*k + 1][2]]

        while i >= len(p):
            i -= len(p)
        ax.scatter(x, y, z, color=p[i])
        k += 1


def bepaalVlakken(x,y,z):

    uiteinden = uiteindenBepalen(x,y,z)
    middelpunten = middelpuntcoordinaten(uiteinden)
    x, y, z = xyz(middelpunten)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    punten(uiteinden, ax)
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    print('De nauwkeurigheid van de benaderde rechte bepaald aan de hand van x en y is ' + str(r))

    y = [a * slope + intercept for a in x]

    yBegin = x[0] * slope + intercept
    yEind = x[-1] * slope + intercept

    lengte = yEind - yBegin
    breedte = x[-1] - x[0]

    lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
    lengtedeelgebieden = lengtegebied / (len(x) - 1)

    line = puntenOpRechte(lengtedeelgebieden, len(x))
    # model = np.poly1d(np.polyfit(line, z, 2))

    fittedParameters = np.polyfit(line, z, 2)
    modelPredictions = np.polyval(fittedParameters, line)
    absError = modelPredictions - z

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE

    print('Een maat voor de nauwkeurigheid van de parabool bedraagt ' + str(RMSE))

    kwadraat = [a * b for a, b in zip(line, line)]
    z = [a * fittedParameters[0] + fittedParameters[1] * b + fittedParameters[2] for a, b in zip(kwadraat, line)]

    ax.plot(x, y, z)

    vlakplot(uiteinden, len(middelpunten), slope, intercept, fittedParameters, ax)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.title("Interpolerende parabool met bijhorende vlakken.")

    plt.show()

    n = 20  # geef aantal middelpunten
    predictparaboolbaanMiddenpunten(middelpunten, n)
    
    predictparaboolbaanUiteindePunten(middelpunten, n)



"""
hoekberekening volgens Bram en Stijn
"""


def filterUitschieters(x, y, z):
    centerz = []
    for index in range(len(z)):
        if index % 2 == 0:
            centerz.append(z[index]+z[index+1])
    mediaan = np.median(centerz)
    newx = []
    newy = []
    newz = []
    for index, center in enumerate(centerz):
        if abs(center-mediaan) < 40:
            newx.append(x[index * 2]), newx.append(x[index * 2 + 1])
            newy.append(y[index * 2]), newy.append(y[index * 2 + 1])
            newz.append(z[index * 2]), newz.append(z[index * 2 + 1])
    return newx, newy, newz


def uiteindenBepalen(x, y, z):                          #zelde functie als bij Jonas
    uiteinden =[]
    for i in range(len(x)):
        uiteinden.append((x[i] ,y[i] ,z[i]))
    return uiteinden


def middelpuntcoordinatenDubbel(uiteinden):
    middelpunten = []

    for i in range(len(uiteinden)):
        if i % 2 == 0:
            x = (uiteinden[i][0] + uiteinden[i + 1][0]) / 2
            y = (uiteinden[i][1] + uiteinden[i + 1][1]) / 2
            z = (uiteinden[i][2] + uiteinden[i + 1][2]) / 2
            middelpunt = (x, y, z)
            middelpunten.append(middelpunt)
            middelpunten.append(middelpunt)
    return middelpunten


def planeFit(coordinaten):
    punten = Points(coordinaten)
    plane = Plane.best_fit(punten)
    #plot_3d(
    #    punten.plotter(c='k', s=50, depthshade=False),
    #    plane.plotter(alpha=0.4, lims_x=(-5, 5), lims_y=(-5, 5)),)
    xs = []
    ys = []
    zs = []
    for coor in coordinaten:
        xs.append(coor[0])
        ys.append(coor[1])
        zs.append(coor[2])

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='b')

    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    # Or use Scipy
    # from scipy.linalg import lstsq
    # fit, residual, rnk, s = lstsq(A, b)
    """
    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("errors: \n", errors)
    print("residual:", residual)
    """
    # plot plane

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                       np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r, c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]
    ax.plot_wireframe(X, Y, Z, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    print(plane.normal)
    return plane.normal


def bepaalHoek(x, y, z):
    newx, newy, newz = filterUitschieters(x, y, z)
    uiteinden = uiteindenBepalen(newx, newy, newz)
    middelpunten = middelpuntcoordinatenDubbel(uiteinden)

    normaalVertical = np.array([0,0,1])
    normaalMiddelpunten = planeFit(middelpunten)
    coorOmOorsprongArray = np.array(uiteinden) - np.array(middelpunten)
    coorOmOorsprong = coorOmOorsprongArray.tolist()
    coorOmOorsprongNorm = []
    for coor in coorOmOorsprong:
        norm = np.sqrt(coor[0]**2 + coor[1]**2 + coor[2]**2)
        coorOmOorsprongNorm.append([coor[0]/norm, coor[1]/norm, coor[2]/norm])



    normaalUiteinden = planeFit(coorOmOorsprongNorm)
    scalairProduct = np.dot(normaalVertical,normaalUiteinden)
    plt.show()
    hoekje = np.arccos(abs(scalairProduct))/np.pi*180
    print(hoekje)
    if hoekje > 45:
        return "Legale worp"
    else:
        return "Illegale worp"




if __name__ == "__main__":
    a, b, c, d = coorVoorBram(vindStok(alleTrajecten1),vindStok(alleTrajecten2))
    listx = []
    listy = []
    listz = []
    for i in range(len(a)):
        puntLinks1 = a[i]
        puntRechts1 = c[i]
        x1, y1, z1 = find_depth(puntLinks1, puntRechts1, B, alpha)
        listx.append(x1)
        listy.append(y1)
        listz.append(z1)
        puntLinks2 = b[i]
        puntRechts2 = d[i]
        x2, y2, z2 = find_depth(puntLinks2, puntRechts2, B, alpha)
        listx.append(x2)
        listy.append(y2)
        listz.append(z2)

    #bepaalVlakken(listx, listz, listy)  #methode Jonas
    """
    of
    """
    print(bepaalHoek(listx, listz, listy)) #methode Bram en Stijn










